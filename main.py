import MySQLdb
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request, session, flash, redirect, send_file
from flask_mail import *
from email.mime.multipart import MIMEMultipart
import smtplib
import pymysql
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import shutil
import io
from flask_mysqldb import MySQL

facedata = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

mydb = pymysql.connect(host='localhost', user='root', password='', port=3306, database='eVote')

sender_address = 'imljs08.04@gmail.com'  # enter sender's email id
sender_pass = 'lifjnercdxvndebu'  # enter sender's password

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hello'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'evote'

mysql = MySQL(app)

with app.app_context():
    def initialize():
        session['IsAdmin'] = False
        session['User'] = None
        session['otp_sent'] = False


@app.route('/')
@app.route('/login')
def login():
    session['otp_sent'] = False
    session['IsAdmin'] = False
    session['User'] = None
    session['otp'] = None
    return render_template('login.html')


@app.route('/admin_dashboard', methods=['POST', 'GET'])
def admin_dashboard():
    return render_template('admin_dashboard.html')


@app.route('/voter_dashboard', methods=['POST', 'GET'])
def voter_dashboard():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT * FROM voters WHERE email=%s', (session['User'],))
    voter_details = cur.fetchone()
    return render_template('voter_dashboard.html', voter_details=voter_details)


@app.route('/admin_login', methods=['POST', 'GET'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if (email == 'admin@voting.com') and (password == 'admin'):
            session['IsAdmin'] = True
            session['User'] = 'admin'
            flash('Admin login successful', 'success')
    return render_template('admin_dashboard.html', admin=session['IsAdmin'])


@app.route('/voter_login', methods=['POST', 'GET'])
def voter_login():
    if request.method == 'POST':
        if request.form['voter_email']:
            email = request.form['voter_email']
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute('SELECT * FROM voters WHERE email=%s', (email,))
            voter = cur.fetchone()
            cur.close()
            if voter:
                if not session['otp_sent']:
                    message = MIMEMultipart()
                    receiver_address = session['email']
                    message['From'] = sender_address
                    message['To'] = email
                    Otp = str(np.random.randint(100000, 999999))
                    print(Otp)
                    session['otp'] = Otp
                    message.attach(MIMEText(session['otp'], 'plain'))
                    abc = smtplib.SMTP('smtp.gmail.com', 587)
                    abc.starttls()
                    abc.login(sender_address, sender_pass)
                    text = message.as_string()  # Converts the message object into a string
                    abc.sendmail(sender_address, receiver_address, text)
                    abc.quit()
                    session['otp_sent'] = True
                    flash('OTP sent to your email', 'success')
                    return render_template('login.html', otp_sent=session['otp_sent'])

        elif request.method == 'GET':
            return render_template('login.html', otp_sent=session['otp_sent'])

        else:
            flash('Invalid Credentials', 'danger')

    return render_template('login.html', otp_sent=session['otp_sent'])


@app.route('/voter_otp_login', methods=['POST', 'GET'])
def voter_otp_login():
    if request.method == 'POST':
        if request.form['voter_otp']:
            voter_otp = request.form['voter_otp']

            if session['otp'] == voter_otp:
                session['IsAdmin'] = False
                session['User'] = session['email']

                cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                cur.execute('SELECT * FROM voters WHERE email=%s', (session['email'],))
                voter = cur.fetchone()
                cur.close()
                print(voter['name'])
                session['name'] = voter['name']
                flash('Voter login successful', 'success')
                return render_template('voter_dashboard.html', voter=session['User'])
            else:
                flash('Invalid OTP', 'danger')

        else:
            flash('Enter OTP', 'danger')

    return render_template('login.html', otp_sent=session['otp_sent'])


@app.route('/admin_logout', methods=['POST', 'GET'])
def admin_logout():
    session['IsAdmin'] = False
    session['User'] = None
    session['otp'] = None
    session['otp_sent'] = False
    flash('Admin Logout Successful', 'success')
    return render_template('login.html')


@app.route('/voter_logout', methods=['POST', 'GET'])
def voter_logout():
    session['IsAdmin'] = False
    session['User'] = None
    session['otp'] = None
    session['otp_sent'] = False
    flash('Voter Logout Successful', 'success')
    return render_template('login.html')


@app.route('/add_voter', methods=['POST', 'GET'])
def add_voter():
    return render_template('add_voter.html')


@app.route('/add_nominee', methods=['POST', 'GET'])
def add_nominee():
    return render_template('add_nominee.html')


@app.route('/view_results', methods=['POST', 'GET'])
def view_results():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT member_name,party_name,vote_count FROM nominee')
    nom = cur.fetchall()
    return render_template('view_results.html', nom=nom)


def get_image_from_db(member_name):
    cur = mysql.connection.cursor()
    cur.execute('SELECT symbol FROM nominee WHERE member_name=%s', (member_name,))
    image_data = cur.fetchone()
    cur.close()
    print(image_data[0])
    return image_data[0] if image_data else None


@app.route('/image/<member_name>')
def image(member_name):
    image_data = get_image_from_db(member_name)
    if image_data:
        return send_file(io.BytesIO(image_data), mimetype='image/jpeg')
    else:
        return "Image not found", 404


def get_image_from_voter(member_name):
    cur = mysql.connection.cursor()
    cur.execute('SELECT photo FROM voters WHERE name=%s', (member_name,))
    image_data = cur.fetchone()
    cur.close()
    print(image_data[0])
    return image_data[0] if image_data else None


@app.route('/imagevoter/<member_name>')
def imagevoter(member_name):
    image_data = get_image_from_voter(member_name)
    if image_data:
        return send_file(io.BytesIO(image_data), mimetype='image/jpeg')
    else:
        return "Image not found", 404


@app.route('/add_nom', methods=['POST', 'GET'])
def add_nom():
    if request.method == 'POST':
        member = request.form['member_name']
        party = request.form['party_name']
        file = request.files['symbol']

        if member and party and file:
            # nominee = pd.read_sql_query('SELECT * FROM nominee', mydb)
            # print(nominee)
            # all_members = nominee.member_name.values
            # all_parties = nominee.party_name.values
            # all_symbols = nominee.symbol.values

            filename = secure_filename(file.filename)
            print(filename)
            img = Image.open(file)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            cur = mysql.connection.cursor()
            cur.execute(
                'INSERT INTO nominee (member_name, party_name, symbol) VALUES (%s, %s, %s)',
                (member, party, img_byte_arr))
            mysql.connection.commit()
            flash(r"Successfully registered a new nominee", 'primary')
        else:
            flash(r"Missing Data", 'danger')
    return render_template('admin_dashboard.html', admin=session['IsAdmin'])


@app.route('/nominee_delete/<variable>', methods=['POST', 'GET'])
def nominee_delete(variable):
    print(variable)
    cur = mysql.connection.cursor()
    cur.execute('DELETE FROM nominee WHERE party_name = %s', (variable,))
    mysql.connection.commit()
    return redirect(url_for('admin_nominee'))


@app.route('/voter_delete/<variable>', methods=['POST', 'GET'])
def voter_delete(variable):
    print(variable)
    cur = mysql.connection.cursor()
    cur.execute('DELETE FROM voters WHERE voter_id = %s', (variable,))
    mysql.connection.commit()
    return redirect(url_for('voter_display'))


@app.route('/nominee', methods=['POST', 'GET'])
def nominee():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT member_name,party_name FROM nominee')
    nom = cur.fetchall()
    return render_template('nominee.html', nom=nom)


@app.route('/admin_nominee', methods=['POST', 'GET'])
def admin_nominee():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT member_name,party_name FROM nominee')
    nom = cur.fetchall()
    return render_template('admin_nominee.html', nom=nom)


@app.route('/voter_display', methods=['POST', 'GET'])
def voter_display():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT * FROM voters')
    voter = cur.fetchall()
    return render_template('voters.html', voter=voter)


# @app.route('/registration', methods=['POST', 'GET'])
# def registration():
#     if request.method == 'POST':
#         first_name = request.form['first_name']
#         last_name = request.form['last_name']
#         state = request.form['state']
#         d_name = request.form['d_name']

#         middle_name = request.form['middle_name']
#         aadhar_id = request.form['aadhar_id']
#         voter_id = request.form['voter_id']
#         pno = request.form['pno']
#         age = int(request.form['age'])
#         email = request.form['email']
#         voters = pd.read_sql_query('SELECT * FROM voters', mydb)
#         print(voters)
#         all_aadhar_ids = voters.aadhar_id.values
#         all_voter_ids = voters.voter_id.values
#         if age >= 18:
#             if (aadhar_id in all_aadhar_ids) or (voter_id in all_voter_ids):
#                 flash(r'Already Registered as a Voter')
#             else:
#                 sql = 'INSERT INTO voters (first_name, middle_name, last_name, aadhar_id, voter_id, email,pno,state,d_name, verified) VALUES (%s,%s,%s, %s, %s, %s, %s, %s, %s, %s)'
#                 cur = mydb.cursor()
#                 cur.execute(sql,
#                             (first_name, middle_name, last_name, aadhar_id, voter_id, email, pno, state, d_name, 'no'))
#                 mydb.commit()
#                 cur.close()
#                 session['aadhar'] = aadhar_id
#                 session['status'] = 'no'
#                 session['email'] = email
#                 return redirect(url_for('verify'))
#         else:
#             flash("if age less than 18 than not eligible for voting", "info")
#     return render_template('voter_reg.html')


@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        voter_id = request.form['voter_id']
        name = request.form['name']
        mobile = request.form['mobile']
        email = request.form['email']
        state = request.form['state']
        d_name = request.form['d_name']
        file = request.files['symbol']
        age = int(request.form['age'])

        if voter_id and name and mobile and email and state and d_name and file and age:

            filename = secure_filename(file.filename)
            print(filename)
            img = Image.open(file)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            voters = pd.read_sql_query('SELECT * FROM voters', mydb)
            all_voter_ids = voters.voter_id.values
            all_emails = voters.email.values

            if age >= 18:
                if voter_id in all_voter_ids or email in all_emails:
                    flash(r'Already Registered as a Voter', 'danger')
                    return render_template('add_voter.html')
                else:
                    sql = 'INSERT INTO voters (voter_id,name,mobile,email,state,district,photo,verified) ' \
                          'VALUES (%s, %s,%s,%s, %s, %s, %s, %s)'
                    cur = mydb.cursor()
                    cur.execute(sql,
                                (voter_id, name, mobile, email, state, d_name, img_byte_arr, 'no'))
                    mydb.commit()
                    cur.close()
                    session['voter_id'] = voter_id
                    session['status'] = 'no'
                    session['email'] = email
                    session['otp_sent'] = False
                    session['capture'] = False
                    session['train'] = False
                    return redirect(url_for('verify'))
            else:
                flash("if age less than 18 than not eligible for voting", "info")
        else:
            flash("Please fill all the fields", "info")
            return render_template('add_voter.html')

    return render_template('add_voter.html')


@app.route('/verify', methods=['POST', 'GET'])
def verify():
    if session['status'] == 'no':
        if request.method == 'POST' and request.form['otp_check']:
            otp_check = request.form['otp_check']
            print(otp_check)
            if otp_check == session['otp']:
                session['status'] = 'yes'
                sql = "UPDATE voters SET verified='%s' WHERE voter_id='%s'" % (session['status'], session['voter_id'])
                cur = mydb.cursor()
                cur.execute(sql)
                mydb.commit()
                cur.close()
                flash(r"Email verified successfully", 'primary')
                return redirect(url_for('capture_images'))  # change it to capture photos
            else:
                flash(r"Wrong OTP. Please try again.", "info")
                return redirect(url_for('verify'))

        else:
            # Sending OTP
            # MIME = Multipurpose Internet Mail Extensions
            if not session['otp_sent']:
                message = MIMEMultipart()
                receiver_address = session['email']
                message['From'] = sender_address
                message['To'] = receiver_address
                Otp = str(np.random.randint(100000, 999999))
                print(Otp)
                session['otp'] = Otp
                message.attach(MIMEText(session['otp'], 'plain'))
                abc = smtplib.SMTP('smtp.gmail.com', 587)
                abc.starttls()
                abc.login(sender_address, sender_pass)
                text = message.as_string()  # Converts the message object into a string
                abc.sendmail(sender_address, receiver_address, text)
                abc.quit()
                session['otp_sent'] = True
    else:
        flash(r"Your email is already verified", 'warning')

    return render_template('verify.html')


@app.route('/voter_verify', methods=['POST', 'GET'])
def voter_verify():
    if request.method == 'POST':
        if request.form['voter_id'] and request.form['voter_email']:
            voter_id = request.form['voter_id']
            voter_email = request.form['voter_email']
            cur = mysql.connection.cursor()
            cur.execute('SELECT voter_id,email FROM voters WHERE voter_id=%s and email=%s', (voter_id, voter_email))
            voter_data = cur.fetchone()
            cur.close()
            if voter_data:
                if not session['voter_otp_sent']:
                    message = MIMEMultipart()
                    receiver_address = session['email']
                    message['From'] = sender_address
                    message['To'] = voter_email
                    Otp = str(np.random.randint(100000, 999999))
                    print(Otp)
                    session['voter_otp'] = Otp
                    message.attach(MIMEText(session['otp'], 'plain'))
                    abc = smtplib.SMTP('smtp.gmail.com', 587)
                    abc.starttls()
                    abc.login(sender_address, sender_pass)
                    text = message.as_string()  # Converts the message object into a string
                    abc.sendmail(sender_address, receiver_address, text)
                    abc.quit()
                    session['voter_otp_sent'] = True
            else:
                flash(r"Please enter Voter ID or Email or Both", 'warning')
        else:
            flash(r"Please enter valid details", 'warning')

    return render_template('login.html')


@app.route('/capture_images', methods=['POST', 'GET'])
def capture_images():
    if request.method == 'POST':
        if not session['capture']:
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 is for camera and CaP_DSHOW is for webcam
            sampleNum = 0  # Number of Images captured
            path_to_store = os.path.join(os.getcwd(), "all_images\\" + session['voter_id'])
            print(path_to_store)
            try:
                shutil.rmtree(path_to_store)
            except:
                pass
            os.makedirs(path_to_store, exist_ok=True)
            while True:
                ret, img = cam.read()
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # turns image into gray
                except:
                    continue
                faces = cascade.detectMultiScale(gray, 1.3, 5)  # scaleFactor=1.3, minNeighbors=5
                for (x, y, w,
                     h) in faces:  # x,y : coordinates of the top left corner of the rectangle , w,h : width and
                    # height of rectangle
                    # draw a rectangle in the center
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),
                                  2)  # (255, 0, 0) is the BGR color of the rectangle , 2 is the thickness of the
                    # rectangle
                    # incrementing sample number
                    sampleNum = sampleNum + 1
                    # saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite(path_to_store + r'\\' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                    # display the frame
                else:
                    cv2.imshow('frame', img)
                    cv2.setWindowProperty('frame', cv2.WND_PROP_TOPMOST, 1)  # to make the window on top
                # wait for 100 miliseconds
                if cv2.waitKey(100) & 0xFF == ord('q'):  # ord('q') is for quitting
                    break
                # break if the sample number is morethan 100
                elif sampleNum >= 200:
                    break
            cam.release()
            cv2.destroyAllWindows()
            flash("Registration is successfully", "success")
            session['capture'] = True
            return redirect(url_for('admin_dashboard'))

        else:
            return render_template('admin_dashboard.html')

    return render_template('capture.html')


from sklearn.preprocessing import LabelEncoder  # for encoding the labels
import pickle  # for saving the model

le = LabelEncoder()


def getImagesAndLabels(path):
    folderPaths = [os.path.join(path, f) for f in
                   os.listdir(path)]  # reads all the folders in the path and creates a list
    print(folderPaths)
    faces = []
    Ids = []
    global le
    for folder in folderPaths:
        imagePaths = [os.path.join(folder, f) for f in os.listdir(folder)]
        aadhar_id = folder.split("\\")[1]
        for imagePath in imagePaths:
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # extract the face from the training image sample
            faces.append(imageNp)
            Ids.append(aadhar_id)
            # Ids.append(int(aadhar_id))
    Ids_new = le.fit_transform(Ids).tolist()  # converting the list to numpy array
    output = open('encoder.pkl', 'wb')  # saving the encoder model , wb is for write binary
    pickle.dump(le, output)
    output.close()
    return faces, Ids_new


# @app.route('/train', methods=['POST', 'GET'])
# def train():
#     if request.method == 'POST':
#         recognizer = cv2.face.LBPHFaceRecognizer_create() # object that compares images
#         faces, Id = getImagesAndLabels(r"all_images")
#         print(Id)
#         print(len(Id))
#         recognizer.train(faces, np.array(Id))  # training the model so that it can recognize the faces
#         recognizer.save("Trained.yml")  # saving the model as Trained.yml
#         flash(r"Model Trained Successfully", 'Primary') 
#         return redirect(url_for('home'))
#     return render_template('train.html')

@app.route('/train')
def train():
    if not session['train']:
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # object that compares images
        faces, Id = getImagesAndLabels(r"all_images")
        print(Id)
        print(len(Id))
        print('Trained')
        recognizer.train(faces, np.array(Id))  # training the model so that it can recognize the faces
        recognizer.save("Trained.yml")  # saving the model as Trained.yml
        flash(r"Model Trained Successfully", 'success')
        session['train'] = True
        return render_template('admin_dashboard.html')
    else:
        flash(r"Model Already Trained", 'danger')
        return render_template('admin_dashboard.html')


# @app.route('/update')
# def update():
#     return render_template('update.html')


# # Route to Update the data of Voters
# @app.route('/updateback', methods=['POST', 'GET'])
# def updateback():
#     if request.method == 'POST':
#         first_name = request.form['first_name']
#         last_name = request.form['last_name']
#         middle_name = request.form['middle_name']
#         aadhar_id = request.form['aadhar_id']
#         voter_id = request.form['voter_id']
#         email = request.form['email']
#         pno = request.form['pno']
#         age = int(request.form['age'])
#         voters = pd.read_sql_query('SELECT * FROM voters', mydb)
#         all_aadhar_ids = voters.aadhar_id.values
#         if age >= 18:
#             if (aadhar_id in all_aadhar_ids):
#                 sql = "UPDATE VOTERS SET first_name=%s, middle_name=%s, last_name=%s, voter_id=%s, email=%s,pno=%s, verified=%s where aadhar_id=%s"
#                 cur = mydb.cursor()
#                 cur.execute(sql, (first_name, middle_name, last_name, voter_id, email, pno, 'no', aadhar_id))
#                 mydb.commit()
#                 cur.close()
#                 session['aadhar'] = aadhar_id
#                 session['status'] = 'no'
#                 session['email'] = email
#                 flash(r'Database Updated Successfully', 'Primary')
#                 return redirect(url_for('verify'))
#             else:
#                 flash(f"Aadhar: {aadhar_id} doesn't exists in the database for updation", 'warning')
#         else:
#             flash("age should be 18 or greater than 18 is eligible", "info")
#     return render_template('update.html')


@app.route('/vote', methods=['POST', 'GET'])
def vote():
    if request.method == 'GET':
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute('SELECT member_name,party_name FROM nominee')
        nom = cur.fetchall()
        return render_template('vote.html', nom=nom)
    else:
        if request.method == 'POST':
            voter_id = session['voter_id']
            party_name = request.form['check']
            cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cur.execute('SELECT * FROM voters WHERE voter_id = % s', (voter_id,))
            account = cur.fetchone()
            if account:
                if account['verified'] == 'yes':
                    cur.execute('SELECT * FROM vote WHERE voter_id = %s', (voter_id,))
                    account = cur.fetchone()
                    if account:
                        flash('You have already voted', 'danger')
                        return redirect(url_for('voter_dashboard'))
                    else:
                        cur.execute('UPDATE nominee SET vote_count=vote_count+1 WHERE party_name=%s', (party_name,))
                        cur.execute('INSERT INTO vote VALUES (%s,%s)', (voter_id, 'yes'))
                        mysql.connection.commit()
                        flash('You have successfully voted', 'success')
                        return redirect(url_for('voter_dashboard'))

                else:
                    flash('Your account is not verified', 'danger')
                    return redirect(url_for('voter_dashboard'))
            else:
                flash('Account does not exists', 'danger')
                return redirect(url_for('voter_dashboard'))

    return render_template('voter_dashboard.html')


# Route for voting
@app.route('/voting', methods=['POST', 'GET'])
def voting():
    if request.method == 'POST':
        pkl_file = open('encoder.pkl', 'rb')  # rb is for read binary
        my_le = pickle.load(pkl_file)  # loading the encoder model
        pkl_file.close()
        recognizer = cv2.face.LBPHFaceRecognizer_create()  # object that compares images
        recognizer.read(
            r"C:\Users\leone\PycharmProjects\eVote\Trained.yml")  # loading the trained model and storing it in
        # recognizer
        cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # capturing the video
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text
        flag = 0
        detected_persons = []
        while True:
            ret, im = cam.read()
            flag += 1
            if flag == 200:
                flash(r"Unable to detect person. Contact help desk for manual voting", "info")
                cv2.destroyAllWindows()
                return render_template('voter_dashboard.html')

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.2,
                                             5)  # detecting the faces, 1.2 is the scale of the image, 5 is the
            # number of neighbors
            print(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])  # predicting the Id and confidence
                print(Id, conf)
                if conf > 40:
                    det_voter = my_le.inverse_transform([Id])[0]
                    detected_persons.append(det_voter)
                    cv2.putText(im, f"Voter ID : {det_voter}", (x, y + h), font, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(im, "Unknown", (x, y + h), font, 1, (255, 255, 255), 2)
            cv2.imshow('im', im)
            try:
                cv2.setWindowProperty('im', cv2.WND_PROP_TOPMOST, 1)
            except:
                pass
            if cv2.waitKey(1) == (ord('q')):
                try:
                    session['select_voter'] = det_voter
                except:
                    cv2.destroyAllWindows()
                    return redirect(url_for('voter_dashboard'))
                cv2.destroyAllWindows()
                return redirect(url_for('vote'))
    return render_template('vote.html')


@app.route('/voter_authenticate', methods=['POST', 'GET'])
def voter_authenticate():
    return render_template('voter_authenticate.html')


# # Route for selecting candidate
# @app.route('/select_candidate', methods=['POST', 'GET'])
# def select_candidate():
#     # extract all nominees
#     aadhar = session['select_aadhar']

#     df_nom = pd.read_sql_query('select * from nominee', mydb)
#     all_nom = df_nom['symbol_name'].values
#     sq = "select * from vote"
#     g = pd.read_sql_query(sq, mydb)
#     all_adhar = g['aadhar'].values
#     if aadhar in all_adhar:
#         flash("You already voted", "warning")
#         return redirect(url_for('home'))
#     else:
#         if request.method == 'POST':
#             vote = request.form['test']
#             session['vote'] = vote
#             sql = "INSERT INTO vote (vote, aadhar) VALUES ('%s', '%s')" % (vote, aadhar)
#             cur = mydb.cursor()
#             cur.execute(sql)
#             mydb.commit()
#             cur.close()
#             # s = "select * from voters where aadhar_id='" + aadhar + "'"
#             # c = pd.read_sql_query(s, mydb)
#             # pno = str(c.values[0][7])
#             # name = str(c.values[0][1])
#             # ts = time.time()
#             # date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#             # timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#             # url = "https://www.fast2sms.com/dev/bulkV2"

#             # # message = 'Hi ' + name + ' You voted successfully. Thank you for voting at ' + timeStamp + ' on ' + date + '.'
#             # no = "7975569230"
#             # message = "helloo hai"
#             # data1 = {
#             #     "route": "q",
#             #     "message": message,
#             #     "language": "english",
#             #     "flash": 0,
#             #     "numbers": no,
#             # }

#             # headers = {
#             #     "authorization": "#",
#             #     "Content-Type": "application/json"
#             # }

#             # response = requests.post(url, headers=headers, json=data1)
#             # print(response)

#             flash(r"Voted Successfully", 'Primary')
#             return redirect(url_for('home'))
#     return render_template('select_candidate.html', noms=sorted(all_nom))


# # Route for voting result
# @app.route('/voting_res')
# def voting_res():
#     votes = pd.read_sql_query('select * from vote', mydb)
#     counts = pd.DataFrame(votes['vote'].value_counts())
#     counts.reset_index(inplace=True)
#     counts.columns = ['img', 'count']
#     print(counts)
#     all_imgs = ['1.png', '2.png', '3.jpg', '4.png', '5.png', '6.png']
#     all_freqs = [counts[counts['img'] == i].iloc[0, 1] if i in counts['img'].values else 0 for i in all_imgs]
#     df_nom = pd.read_sql_query('select * from nominee', mydb)
#     all_nom = df_nom['symbol_name'].values
#     return render_template('voting_res.html', freq=all_freqs, noms=all_nom)


if __name__ == '__main__':
    app.run(debug=True)
