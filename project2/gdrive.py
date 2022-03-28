from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
drive = GoogleDrive(gauth)

root_id = "1o0FPKu2zu4NqvjxnCrxq6ZyHNogTwNah"

# upload_file_list = ["mcts.py", "rl.py"]
# for upload_file in upload_file_list:
#     gfile = drive.CreateFile({"parents": [{"id": "1o0FPKu2zu4NqvjxnCrxq6ZyHNogTwNah"}]})
#     # Read file and set it as the content of this instance.
#     gfile.SetContentFile(upload_file)
#     gfile.Upload()  # Upload the file.

file_metadata = {
    "name": "test_folder",
    "title": "testing",
    "parents": [{"id": root_id}],
    "mimeType": "application/vnd.google-apps.folder",
}

folder = drive.CreateFile(file_metadata)
folder.Upload()

upload_file_list = ["mcts.py", "rl.py"]
for upload_file in upload_file_list:
    gfile = drive.CreateFile({"parents": [{"id": folder["id"]}]})
    # Read file and set it as the content of this instance.
    gfile.SetContentFile(upload_file)
    gfile.Upload()  # Upload the file.

