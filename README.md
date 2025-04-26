# Detect_Toxic_Cmt_VN
<br>
This model will help detect every toxic comment on social media (HeritaHub)


This model is base on:<br>

Nhattan040102. (2023). Vietnamese Hate and Offensive Detection using PhoBERT, CNN, and Social Media Streaming Data [Computer software]. GitHub. https://github.com/nhattan040102/Vietnamese-Hate-and-Offensive-Detection-using-PhoBERT-CNN-and-Social-Media-Streaming-Data (Accessed April 14, 2025)<br>
This project uses the pretrained PhoBERT-CNN model provided by Nhattan040102 on GitHub for Vietnamese hate and offensive speech (text) detection. <br>
I implement this model to host a server api to detect any comment using Vietnamese language that recently post.<br>
I choose Ngrok to host server with a static domain : "ngrok http --url=crab-enjoyed-buck.ngrok-free.app 5000"<br>


Tutorial:<br>
1. Git clone this repository to your folder "git clone https://github.com/truongNgn/Detect_Toxic_Cmt_VN.git"
2. There is a requirement.txt, you need to install of all its library to run this model and server.
3. Install ngrok via Chocolatey with the following command: "choco install ngrok"
4. Run the command to add your authtoken to the default ngrok.yml configuration file 
5. Open cmd to host server, paste this to cmd ""ngrok http --url=crab-enjoyed-buck.ngrok-free.app 5000"", this will host a local server to run api
6. Open and run app.py, this will run the model and process any comment recently post from user.

*Note:
The api return <br>
'2' : Tích cực<br>
'1' : Trung lập<br>
'-1': Tiêu cực<br>
