from encord import EncordUserClient
from encord.metadata_schema import MetadataSchema
import os
from encord.orm.dataset import LongPollingStatus
from encord.storage import FoldersSortBy

key_path = os.environ.get("ENCORD_SSH_KEY_FILE")

user_client: EncordUserClient = EncordUserClient.create_with_ssh_private_key(
    ssh_private_key_path=key_path
)

# # Create the schema
# metadata_schema = user_client.metadata_schema()
# metadata_schema.add_scalar("id", data_type="text")
# metadata_schema.save()


# Specify the integration you want to use by replacing <integration_title> with the integration title
integrations = user_client.get_cloud_integrations()
integration_idx = [i.title for i in integrations].index("docs media")
integration = integrations[integration_idx].id





# Find the storage folder by name
folder_name = "Encord Documentation Images"  # Replace with your folder's name
folders = list(user_client.find_storage_folders(search=folder_name, dataset_synced=None, order=FoldersSortBy.NAME, desc=False, page_size=1000))

# Ensure the folder was found
if folders:
    storage_folder = folders[0]

    # Initiate cloud data registration to the storage folder. Replace path/to/json/file.json with the path to your JSON file
    upload_job_id = storage_folder.add_private_data_to_folder_start(
        integration_id=integration, private_files="doc_images.json", ignore_errors=True
    )

    # timeout_seconds determines how long the code waits after initiating upload before checking the upload status
    res = storage_folder.add_private_data_to_folder_get_result(upload_job_id, timeout_seconds=5)
    print(f"Execution result: {res}")

    if res.status == LongPollingStatus.PENDING:
        print("Upload is still in progress, try again later!")
    elif res.status == LongPollingStatus.DONE:
        print("Upload completed")
        if res.unit_errors:
            print("The following URLs failed to upload:")
            for e in res.unit_errors:
                print(e.object_urls)
    else:
        print(f"Upload failed: {res.errors}")
else:
    print("Folder not found")