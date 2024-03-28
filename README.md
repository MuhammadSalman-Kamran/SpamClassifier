# SpamClassifier

## This is end-to-end SMS/Email Spam Classifier

## Steps to deploy streamlit app on AWS EC2

1. Login with your AWS console and launch an EC2 instance

2. Run the following commands

Note: Do the port mapping to this port:- 8501

```bash
sudo apt update
```

```bash
sudo apt-get update
```

```bash
sudo apt upgrade -y
```

```bash
sudo apt install git curl unzip tar make sudo vim wget -y
```

```bash
git clone "Your-repository"
```

```bash
sudo apt install python3-pip
```

```bash
pip3 install -r requirements.txt
```

```bash
#Temporary running
python3 -m streamlit run app.py
```

```bash
#Permanent running
nohup python3 -m streamlit run app.py
```

Note: Streamlit runs on this port: 8501

URL of the project : 44.222.201.199:8501

![alt text](https://github.com/MuhammadSalman-Kamran/SpamClassifier/blob/main/running_app.jpg)
