# 1단계: AWS EC2 인스턴스 생성 및 보안 설정

1. AMI 선택: **Ubuntu Server 22.04 LTS (HVM)**를 선택합니다. (Python 3.10 기본 탑재)
2. 인스턴스 유형: t3.micro를 선택합니다.
3. 보안 그룹(Security Group) 설정: 인바운드 규칙에 다음을 추가해야 외부에서 API에 접속할 수 있습니다.
    - SSH: TCP, 22번 포트 (내 IP)
    - Custom TCP: TCP, 8000번 포트 (Anywhere 0.0.0.0/0) — FastAPI의 기본 포트입니다.

# 2단계: 서버 접속 및 기본 환경 설정

터미널(또는 CMD)에서 생성한 키페어(.pem)를 이용해 서버에 접속합니다.

```
# 1. 키페어 권한 변경 (Linux/Mac 사용자만 해당)
chmod 400 your-key.pem

# 2. SSH 접속
ssh -i "your-key.pem" ubuntu@<인스턴스-퍼블릭-IP>
```

접속 후 패키지 업데이트 및 필수 도구를 설치합니다.

```
sudo apt update && sudo apt upgrade -y
# Python 3.10은 이미 설치되어 있으므로 pip와 venv만 설치합니다.
sudo apt install -y python3-pip python3-venv git
```

# 3단계: GitHub 연동을 위한 SSH Key 등록

1. EC2에서 SSH 키 생성
   먼저 EC2 터미널에서 GitHub과 통신할 때 사용할 열쇠(SSH Key)를 만듭니다.

```
# 1. SSH 키 생성. 중간 옵션들은 전부 엔터를 칩니다.
ssh-keygen

# 2. SSH 키 확인
cat ~/.ssh/id_rsa.pub
# 이제 생성된 **공개 키(Public Key)**의 내용을 확인하고 복사합니다.
```

2. GitHub 저장소에 Deploy Key 등록
   복사한 키를 GitHub의 해당 Private 저장소에 알려줘야 합니다.

-   GitHub의 해당 **Private 저장소(Repository)**로 이동합니다.
-   상단 메뉴의 Settings 클릭.
-   왼쪽 사이드바에서 Deploy keys 메뉴를 클릭합니다.
-   오른쪽 상단의 Add deploy key 버튼을 클릭합니다.
-   Title: EC2-FastAPI-Server (본인이 알아보기 쉬운 이름)
-   Key: 아까 터미널에서 복사한 문자열을 붙여넣습니다.
-   (중요) 서버에서 코드 수정 후 git push까지 할 계획이라면 Allow write access를 체크하세요. (단순 배포용이면 체크 안 해도 됩니다.)
-   Add key 버튼을 눌러 저장합니다.

# 4단계: GitHub 연동 및 코드 배포

1. 코드 가져오기
   GitHub 저장소에서 코드를 클론합니다. (Public 저장소 기준이며, Private인 경우 SSH Key 등록이 필요할 수 있습니다.)

```
cd /home/ubuntu
# private Github repository에 연결하는 것이므로 ssh clone을 합니다.
git clone git@github.com:사용자이름/저장소이름.git
cd 저장소이름
```

2. 가상환경 설정 및 라이브러리 설치
   시스템 라이브러리와 충돌을 방지하기 위해 가상환경을 사용합니다.

```
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 필수 라이브러리 설치
pip install fastapi uvicorn
# 만약 requirements.txt가 있다면
# pip install -r requirements.txt
```

# 5단계: FastAPI 서버 상시 가동 설정 (systemd)

단순히 uvicorn 명령어를 실행하면 터미널을 종료할 때 서버도 꺼집니다. 서버를 백그라운드에서 상시 구동하기 위해 systemd 서비스로 등록합니다.

1. 서비스 파일 생성

```
sudo vim /etc/systemd/system/fastapi.service
```

2. 내용 작성 (아래 내용을 복사해서 붙여넣으세요)
   WorkingDirectory와 ExecStart의 경로는 본인의 프로젝트명에 맞게 수정해야 합니다.

```
[Unit]
Description=Gunicorn instance to serve FastAPI
After=network.target

[Service]
User=ubuntu
Group=www-data
# 프로젝트 폴더 경로
WorkingDirectory=/home/ubuntu/저장소이름
# 가상환경 내 uvicorn 실행 경로
ExecStart=/home/ubuntu/저장소이름/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

3. 서비스 시작 및 활성화

```
# 데몬 재로드
sudo systemctl daemon-reload

# 서비스 시작 및 부팅 시 자동 실행 설정
sudo systemctl start fastapi
sudo systemctl enable fastapi

# 상태 확인
sudo systemctl status fastapi
```

# 6단계: 동작 확인

이제 브라우저나 API 테스트 도구(Postman 등)에서 아래 주소로 접속하여 서버가 잘 동작하는지 확인합니다.

http://<인스턴스-퍼블릭-IP>:8000

# 7단계: Nginx 연동 설정 방법 (실전 코드)

1. Nginx 설치 및 기본 설정 파일 생성

```
sudo apt install nginx -y

# 기존 기본 설정 삭제 후 새로 생성
sudo rm /etc/nginx/sites-enabled/default
sudo vim /etc/nginx/sites-available/fastapi
```

2. Nginx 설정 내용 작성
   아래 내용을 복사해서 붙여넣으세요. 이 설정은 80번 포트로 들어온 모든 요청을 내부의 8000번(FastAPI)으로 넘겨줍니다.

```
server {
    listen 80;
    server_name <본인의-EC2-퍼블릭-IP>;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

3. 설정 활성화 및 재시작

```
# 설정 파일 연결
sudo ln -s /etc/nginx/sites-available/fastapi /etc/nginx/sites-enabled/

# Nginx 설정 문법 검사
sudo nginx -t

# Nginx 재시작
sudo systemctl restart nginx
```

# 8단계: 동작 확인

Nginx를 통해 80번 포트로 서비스를 열었으므로, AWS 콘솔에서도 해당 포트를 열어주어야 합니다.

1. AWS EC2 콘솔 -> 보안 그룹(Security Groups) 이동.
2. 사용 중인 보안 그룹 선택 후 인바운드 규칙 편집(Edit inbound rules) 클릭.
3. HTTP (TCP 80) 규칙을 추가 (Source: 0.0.0.0/0).
4. (선택 사항) 이제 외부에서 8000번으로 직접 들어올 필요가 없다면, 기존 8000번 규칙은 삭제하여 보안을 더 강화할 수 있습니다.

결과: 이제 브라우저 주소창에 포트 번호 없이 http://<EC2-IP>만 입력해도 FastAPI 서버와 통신이 가능해집니다!
