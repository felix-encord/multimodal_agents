
from encord.objects.classification import Classification
from encord.objects.classification_instance import ClassificationInstance
from encord.objects.ontology_labels_impl import LabelRowV2
from openai import OpenAI
from pydantic import ValidationError
import json
from encord_agents.core.data_model import Frame, LabelRowMetadataIncludeArgs
from encord_agents.core.ontology import OntologyDataModel
from encord_agents.tasks import Runner

from typing import Annotated

from encord import EncordUserClient

from encord_agents.core.utils import download_asset


from encord_agents.tasks import Depends
from encord_agents.tasks.dependencies import dep_single_frame, dep_asset
from pathlib import Path
from typing import Annotated
import os
from encord.objects.classification import Classification
from encord.objects.classification_instance import ClassificationInstance
from encord.objects.ontology_labels_impl import LabelRowV2
from typing_extensions import Annotated
from encord_agents.core.data_model import Frame
from encord_agents.tasks import Depends, Runner
import numpy as np
import csv

PROJECT_HASH = "41c9b219-141b-488b-804f-2991692a5d6a"
runner = Runner(project_hash=PROJECT_HASH)
user_client = EncordUserClient.create_with_ssh_private_key(ssh_private_key_path = os.environ.get('ENCORD_SSH_KEY_FILE'))
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
classifications = runner.project.ontology_structure.classifications
ont_data_model: OntologyDataModel[Classification] = OntologyDataModel(classifications)



def prompt_image_input_to_openai(prompt : str,frame : Frame):

    b64_frame = frame.b64_encoding(output_format="openai")
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, b64_frame ], 
            }
        ],
        response_format={"type": "json_object"},
    )

    model_response = response.choices[0].message.content or "Failed to get resp"

    return model_response



def generate_qa_candidates(
    frame : Frame,
    above_text : str,
    below_text : str
) -> list[ClassificationInstance] :
    
    prompt = f"""
You are tasked with generating a visual question-answer dataset for training a multimodal assistant. The assistant sees the user's screen when they interact with our SaaS product and answers user questions based on the visible context. 

To help you create this dataset you are provided with a screenshot in the product's documentation and its surrounding text:

- Documentation text immediately **above** the screenshot:
{above_text}

- Documentation text immediately **below** the screenshot:
{below_text}

### Task Instructions:
1. Thoroughly analyze the provided screenshot, imagining you are a user interacting with the application as depicted.
2. Generate **three realistic, diverse questions** a user might ask in this scenario. Ensure at least one question is framed as a "next-step" question, such as "What do I do next to achieve [a specific task]?"
3. For each question, provide an accurate and concise answer explicitly referencing visual details from the screenshot. If possible provide information from the documentation in your answer - but if the documentation is insufficient then derive answers directly from the screenshot.
4. Ensure all questions and answers explicitly reference visual details from the screenshot.

### Response Format:
Respond strictly using the following JSON Schema.
{{
    "QA pair 1 : {{
    "Question" : <question>,
    "Answer" : <answer>,
    }},
    "QA pair 2 : {{
    "Question" : <question>,
    "Answer" : <answer>,
    }},
    "QA pair 3 : {{
    "Question" : <question>,
    "Answer" : <answer>,
    }}  
}}
"""

    candidate_qas = prompt_image_input_to_openai(prompt,frame)
    # candidate_qas = "FElix"

    filled_ont_schema = map_agent_output_to_schema(candidate_qas)
    
    try:
        instances = ont_data_model(json.dumps(filled_ont_schema))
    
    except ValidationError as e:
        # invalid json
        print("Failed to get appropriate JSON Schema resp")
        print(f"{candidate_qas=}")
        print(f"{ont_data_model.model_json_schema_str=}")
        instances = []
    
    return instances
    
def set_qa_candidates(lr: LabelRowV2, frame: Frame, above_text: str, below_text: str) -> None:
  

    classification_instances = generate_qa_candidates(frame,above_text,below_text)

    # Store the answers in the label row
    for inst in classification_instances:
        inst.set_for_frames(overwrite=True)
        lr.add_classification_instance(inst,force=True)


def map_agent_output_to_schema(candidate_qas):

    candidate_qas_json = json.loads(candidate_qas)
    
    empty_ont = {
    "question_answer_pair_1": {
        "feature_node_hash": "D9oJlIXz",
        "choice": {
            "feature_node_hash": "5qvmjZEN",
            "question": {
                "feature_node_hash": "L27fSN3D",
                "value": "NOTFILLEDIN"
            },
            "answer": {
                "feature_node_hash": "4mvdzZ+c",
                "value": "NOTFILLEDIN"
            }
        }
    },
    "question_answer_pair_2": {
        "feature_node_hash": "jat0dizi",
        "choice": {
            "feature_node_hash": "7ew0GaDG",
            "question": {
                "feature_node_hash": "yZdu3EEi",
                "value": "NOTFILLEDIN"
            },
            "answer": {
                "feature_node_hash": "RtwPHjxE",
                "value": "NOTFILLEDIN"
            }
        }
    },
    "question_answer_pair_3": {
        "feature_node_hash": "cZsafTBg",
        "choice": {
            "feature_node_hash": "FVgOeelB",
            "question": {
                "feature_node_hash": "8IpLiz/c",
                "value": "NOTFILLEDIN"
            },
            "answer": {
                "feature_node_hash": "PV+lOe7L",
                "value": "NOTFILLEDIN"
            }
        }
    }
}
    for i,qa in enumerate(candidate_qas_json.keys()):
        
        qa_index = i+1

        q =  candidate_qas_json[qa]['Question']
        a = candidate_qas_json[qa]['Answer']


        empty_ont[f'question_answer_pair_{qa_index}']['choice']['question']['value'] = q
        empty_ont[f'question_answer_pair_{qa_index}']['choice']['answer']['value'] = a
    
    return empty_ont

def make_lr_db()-> tuple[dict[str,LabelRowV2]]:
    project = user_client.get_project(PROJECT_HASH)
    label_rows = project.list_label_rows_v2(include_client_metadata=True)

    global above_text_db 
    global below_text_db 

    above_text_db = {}
    below_text_db = {}
    for label_row in label_rows:
        client_metadata = label_row.client_metadata
        if client_metadata['Data_Type'] == 'above_text':
            above_text_db[client_metadata['id']] = label_row
        if client_metadata['Data_Type'] == 'below_text':
            below_text_db[client_metadata['id']] = label_row
    
def dep_get_above_text(lr : LabelRowV2)-> str:
    
    id = lr.client_metadata['id']
    text_lr = above_text_db[id]
    # data_hash = above_text_db[id].data_hash
    with download_asset(text_lr) as asset_fp:
   
        text_contents = asset_fp.read_text()
    
    return text_contents

def dep_get_below_text(lr : LabelRowV2)->str:
    
    id = lr.client_metadata['id']
    text_lr = below_text_db[id]
    # data_hash = above_text_db[id].data_hash
    with download_asset(text_lr) as asset_fp:
        text_contents = asset_fp.read_text()
    
    return text_contents
            
args = LabelRowMetadataIncludeArgs(
    include_client_metadata=True,
)

@runner.stage(stage = 'QA Generation',label_row_metadata_include_args=args)
def agent_make_qa_candidates(
    lr : LabelRowV2,
    above_text : Annotated[str,Depends(dep_get_above_text)],
    below_text : Annotated[str,Depends(dep_get_below_text)],
):
    
    metadata = lr.client_metadata
    
    if metadata['Data_Type'] != 'Image':
        lr.save()
        return "To Review"
    
    frame = Frame(frame=0,content=dep_single_frame(lr))
    lr.initialise_labels(overwrite=True)
    set_qa_candidates(lr,frame,above_text=above_text,below_text=below_text)
    lr.save()

    return "To Review"

'''run this to generate qa pairs'''
# make_lr_db()
# runner()

''' run this after reviewing qa pairs to save to csv file (index matches up with original image-docs dataset'''

caption_file = Path('qa_results_2.csv')
has_content = caption_file.is_file()

if not has_content:
    with caption_file.open("a" if has_content else "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        cols = ['index','Question 1','Answer 1','Question 2','Answer 2','Question 3','Answer 3']
        writer.writerow(cols)  # Writes the header

data_transfer_runner = Runner(PROJECT_HASH)

@data_transfer_runner.stage(stage='Save Results',label_row_metadata_include_args=args)
def transfer_data(
    lr: LabelRowV2, 
) -> str:

    metadata = lr.client_metadata


    id = metadata['id']
  

    if metadata['Data_Type'] != 'Image':
        lr.save()
        return "transferred"
    
    root = Path('qa_pairs')
    instances = lr.get_classification_instances()

    qlist = []
    alist = []
    qa_set = set()
    for i,instance in enumerate(sorted(instances, key = lambda x: x.classification_name)):
        
        for ans in instance.get_all_static_answers():
            ans = ans.to_encord_dict()

            # due to latency issue some LRs were input into agent twice, so need to skip double inputs
            if 'Question Answer Pair' in ans['name']:
                if ans['name'] in qa_set:
                    continue
                else:
                    qa_set.add(ans['name'])

            if ans['name'] == 'Question':
                qlist.append(ans['answers'])
            if ans['name'] == 'Answer':
                alist.append(ans['answers'])

    write_list = [id]
    for q,a in zip(qlist,alist):
        write_list.append(q)
        write_list.append(a)
    
    if len(write_list) == 7:
        print(f'writing {id}')
        with caption_file.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(write_list)



    return "transferred"

data_transfer_runner()

