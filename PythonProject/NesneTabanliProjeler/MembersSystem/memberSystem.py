import json
from random import randint
import re
import hashlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

class UserDataService:
    def __init__(self, file_path="users.json"):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.file_path, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        if isinstance(data, list):
            data = {"users": data}  # listeyi 'users' anahtarının altına koy
        elif "users" not in data:
            data["users"] = []
        
        self.save_data(data)
        return data
    
    def save_data(self, data=None):
        if data is None:
            data = self.data
        with open(self.file_path, "w") as file:
            json.dump(data, file, indent=4)

    def get_all_users(self):
        return self.data["users"]
    
    def find_user_by_username(self, username):
        for user in self.data["users"]:
            if user ["username"] == username:
                return user
        return None

    def find_user_by_email(self, email):
        for user in self.data["users"]:
            if user["email"] == email:
                return user
        return None   

    def update_user(self, updated_user):
        for i, user in enumerate(self.data["users"]):
            if user["username"] == updated_user["username"]:
                self.data["users"][i] = updated_user
                self.save_data()
                return

    def add_user(self, user_dict):
        self.data["users"].append(user_dict)
        self.save_data()   


class EmailService:
    def __init__(self, sender_email, sender_password):
        self.sender_email = sender_email
        self.sender_password = sender_password

    def send_activation_email(self, to_email, activation_code):
        subject = "Activation Code for Membership System"
        body = f"Your activation code is: {activation_code}" 

        message = MIMEMultipart()
        message["Form"] = self.sender_email
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(message)
            server.quit()
            print("✅ Activation code has been sent to your email.")
        except Exception as e:
            print("❌ Email sending failed:", e)

class System:
    def __init__(self):
        self.systemState = True
        self.data_service = UserDataService()
        self.email_service = EmailService(
            sender_email="adresinigir@gmail.com",
            sender_password="uygulamaşifresial"
        )
        

    def run(self):
        self.showMenu()
        select = self.selectMenu()

        if select == 1:
            self.login()
        if select == 2:
            self.register()
        if select == 3:
            self.forgotPassword()
        if select == 4:
            self.exitSystem()
        
    def showMenu(self):
        print("\n*********** Welcome to the Membership System ***********")
        print("1. Login")
        print("2. Register")
        print("3. Forgot Password")
        print("4. Exit")

    def selectMenu(self):
        while True:
            try:
                select = int(input("Enter your choice: "))
                while select < 1 or select > 4:
                    select = int(input("Please choose between 1 and 4: "))
                return select
            except ValueError:
                print("Please enter a number!!!")


    def login(self):
        username = input("Enter your username: ")
        password = input("Enter your password: ")
        
        hashedPass = self.hashPassword(password)
        systemState = self.checkCredentials(username, hashedPass)

        if systemState:
            self.loginSuccess()
        else:
            self.loginFailed()

    def checkCredentials(self, username, password):
        users = self.data_service.get_all_users()
        for user in users:
            if user["username"] == username:
                if user["timeout"]:
                    timeout_time = datetime.strptime(user["timeout"], "%Y-%m-%d %H:%M:%S")
                    if datetime.now() < timeout_time:
                        print("🚫 Too many failed attempts. Please try again later.")
                        return False
                    else:
                        user["timeout"] = ""

                if user["password"] == password and user["Activation"] == "Y":
                    user["attempt"] = 0
                    self.data_service.save_data(users)
                    return True
                else:
                    user["attempt"] = user.get("attempt", 0) + 1
                    if user["attempt"] >= 3:
                        user["timeout"] = (datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
                        user["attempt"] = 0
                        print("🚫 3 failed attempts. You are blocked for 5 minutes.")
                    self.data_service.save_data(users)
                    return False
        return False

    
    def loginFailed(self, reason="The information entered is incorrect"):
        print(reason)

    def loginSuccess(self):
        print("Welcome to Our System!")
        self.systemState = False

    def register(self):
        username = input("Enter your username: ")
        
        while True:
            password = input("Enter your password: ")
            passAgain = input("Re-enter your password: ")

            if password == passAgain:
                break
            else:
                print("Passwords do not match. Please enter again.")

        while True:
            email = input("Please enter your email address: ").strip()

            if not self.is_valid_email(email):
                print("❌ Please enter a valid email address (example: abc@example.com)")
                continue

            if self.isEmailRegistered(email):
                print("❌ This email is already registered in the system!")
                return  # kayıt işlemini iptal ediyoruz

            break  # her şey doğruysa çık

        activationCode = self.sendActivationCode(email)
        activationState = self.verifyActivationCode(activationCode)

        if activationState:
            hashedPass = self.hashPassword(password)
            self.saveUser(username, hashedPass, email)
        else:
            print("Activation is invalid!")



    def forgotPassword(self):
        email = input("Please enter your email address: ")

        if self.isEmailRegistered(email):
            activation = self.sendActivationCode(email)  # Email gönder

            activationLogin = input("Enter the activation code we sent to change your password: ").strip()

            if activationLogin == activation:
                while True:
                    newPass = input("Enter your new password: ")
                    newPassA = input("Re-enter your new password: ")

                    if newPass == newPassA :
                        break
                    else:
                        print("The passwords you entered do not match, please enter again.")

                users = self.data_service.get_all_users()

                for user in users:
                    if user["email"] == email:
                        user["password"] = self.hashPassword(newPass)
                self.data_service.save_data(users)
                print("Password changed successfully!")
            else:
                print("Activation code is incorrect!")
        else:
            print("Such an email is not registered in our system!")

    def isEmailRegistered(self, email):
        return self.data_service.find_user_by_email(email) is not None

    def exitSystem(self):
        self.systemState = False

    def isAccountRegistered(self, username, email):
        user = self.data_service.find_user_by_username(username)

        if user and user["email"] == email:
            return True
        return False

    def sendActivationCode(self, email):
        activation = str(randint(1000, 9999))
        self.email_service.send_activation_email(email, activation)
        return activation

    def verifyActivationCode(self, activation):
        getActivationCode = input("Enter your activation code: ").strip()
        
        if activation == getActivationCode:
            return True
        else:
            return False
        
    def saveUser(self, username, password, email):
        users = self.data_service.get_all_users()
        users.append({
            "username": username,
            "password": password,
            "email": email,
            "Activation": "Y",
            "timeout": "",
            "attempt": 0
        })
        self.data_service.save_data(users)
        print("Record created successfully!")

    def hashPassword(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def is_valid_email(self, email):
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        return re.match(pattern, email) is not None


system = System()

while system.systemState:
    system.run()


