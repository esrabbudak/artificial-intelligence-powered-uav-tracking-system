import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText  

def sendMail(toMail, subject, content):
    fromMail = "gönderen mail adresi"
    password = "mail adresin uygulama şifresi edinip buraya eklenmeli"  # Uygulama özel şifresi

    # SMTP bağlantısı
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(fromMail, password)

    # Mail oluşturma
    message = MIMEMultipart('alternative')
    message['Subject'] = subject
    message['From'] = fromMail
    message['To'] = toMail

    # HTML içerik ekleme
    htmlContent = MIMEText(content, 'html')
    message.attach(htmlContent)

    # Mail gönderme
    server.sendmail(fromMail, toMail, message.as_string())
    server.quit()
