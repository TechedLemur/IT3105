from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

gauth = GoogleAuth()
drive = GoogleDrive(gauth)

root_id = "1o0FPKu2zu4NqvjxnCrxq6ZyHNogTwNah"


def create_folder(parent_id, folder_name):
    folder_metadata = {
        "name": folder_name,
        "title": folder_name,
        "parents": [{"id": parent_id}],
        "mimeType": "application/vnd.google-apps.folder",
    }
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    return folder["id"]


def add_files_to_folder(path, folder_id):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            print("Uploading file: ", file)
            gfile = drive.CreateFile({"title": file, "parents": [{"id": folder_id}]})
            gfile.SetContentFile(f"{path}/{file}")
            gfile.Upload()  # Upload the file.


def upload_data(folder_name, path="data"):
    data_folder_id = create_folder(root_id, folder_name)
    dataset_folder_id = create_folder(data_folder_id, "dataset")
    model_folder_id = create_folder(data_folder_id, "models")

    add_files_to_folder(f"{path}/{folder_name}", data_folder_id)
    add_files_to_folder(f"{path}/{folder_name}/models", dataset_folder_id)
    add_files_to_folder(f"{path}/{folder_name}/dataset", model_folder_id)


if __name__ == "__main__":
    upload_data("2022-03-27T19-41-30_7x7")
