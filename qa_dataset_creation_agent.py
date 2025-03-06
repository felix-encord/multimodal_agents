# Imports
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal

from encord.constants.enums import DataType
from encord.http.bundle import Bundle
from encord.objects.classification import Classification
from encord.objects.classification_instance import ClassificationInstance
from encord.objects.ontology_labels_impl import LabelRowV2
from openai import OpenAI
from pydantic import ValidationError

from encord_agents.core.data_model import Frame, LabelRowMetadataIncludeArgs
from encord_agents.core.ontology import OntologyDataModel
from encord_agents.tasks import Runner
from encord_agents.tasks.dependencies import dep_asset
from collections import defaultdict
from typing import Annotated

from encord import EncordUserClient
from encord.dataset import Dataset
from encord.orm.dataset import DataLinkDuplicatesBehavior
from encord.project import Project
from encord_agents.core.utils import download_asset
from encord.storage import StorageItem

from encord_agents.tasks import Depends
from encord_agents.tasks.dependencies import dep_client, dep_single_frame, dep_storage_item
import tempfile
from typing import Annotated
from uuid import uuid4
import os
from encord.workflow.stages.agent import AgentStage
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from typing import Iterable

from encord.objects.classification import Classification
from encord.objects.classification_instance import ClassificationInstance
from encord.objects.ontology_labels_impl import LabelRowV2
from encord.project import Project
from typing_extensions import Annotated

from encord_agents.core.data_model import Frame
from encord_agents.tasks import Depends, Runner
import numpy as np

PROJECT_HASH = "5db00287-08fc-4381-9752-747ecb367f13"
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
                "content": [{"type": "text", "text": prompt} ],
            }
        ],
    )
    model_response_parent = response.choices[0].message
    model_response = response.choices[0].message.content or "Failed to get resp"

    return model_response



def generate_qa_candidates(
    frame : Frame,
    above_text : str,
    below_text : str
) -> list[ClassificationInstance] :
    
    prompt = f"""Help me create a question-answer dataset for my application. You have access to a screenshot of the application from the documentation and the surrounding text above and below it:

    === documentation text above image ===
    {above_text}

    === documentation text below image ===
    {below_text}

    ==== instructions ===
    Thoroughly review the screenshot, imagining you are a user interacting with the application in a context similar to what is shown. Based on that situation, propose three realistic questions that such a user might ask, and then provide accurate answers to each question based on the documentation text. If the documentation text is not informative enough, come up with question-answer pairs using the screenshot only. Ensure the questions are realistic but sufficiently diverse. The questions and answers ***must*** refer to the image.

    Please follow the JSON Schema to indicate your response.
    Don't respond with anything but valid json.

    === JSON Schema ===
    {ont_data_model.model_json_schema_str}
    """

    candidate_qas = prompt_image_input_to_openai(prompt,frame)

    try:
        instances = ont_data_model(candidate_qas)
    except ValidationError as e:
        # invalid json
        print("Failed to get appropriate JSON Schema resp")
        print(f"{candidate_qas=}")
        print(f"{ont_data_model.model_json_schema_str=}")
        raise e
    
    return instances
    
def set_qa_candidates(lr: LabelRowV2, frame: Frame, above_text: str, below_text: str) -> None:
  

    classification_instances = generate_qa_candidates(frame,above_text,below_text)

    # Store the answers in the label row
    for inst in classification_instances:
        inst.set_for_frames()
        lr.add_classification_instance(inst)



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
    img : Annotated[Frame, Depends(dep_single_frame)],
    above_text : Annotated[str,Depends(dep_get_above_text)],
    below_text : Annotated[str,Depends(dep_get_below_text)],
):
    
    metadata = lr.client_metadata
    
    if metadata['Data_Type'] != 'Image':
        return "To Review"
    
    frame = Frame(frame=0,content=img)
    lr.initialise_labels(overwrite=True)
    set_qa_candidates(lr,frame,above_text=above_text,below_text=below_text)

    return "To Review"


make_lr_db()
runner()
