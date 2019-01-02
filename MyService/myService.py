import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


class MyService :
    def __init__(self,numero_salle,numero_table):
        cred = credentials.Certificate(
            "/home/myadmin/PycharmProjects/testFirebase/digitalinnovation-69f64-firebase-adminsdk-uy3pm-91cead3889.json");
        default_app = firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://digitalinnovation-69f64.firebaseio.com/'
        })
        self.mainRef = db.reference().child("salles").child(str(numero_salle)).child("tables").child(str(numero_table))
        try :
            self.mainRef.set("")
        except :
            print("exception")


    def update_face(self,nom, occurences):
        try :
            self.mainRef.child("recognized_faces").child(nom).set(occurences)
        except :
            print("exception")
    def update_object(self,nom, occurences):
        try :
            self.mainRef.child("detected_objects").child(nom).set(occurences)
        except :
            print("exception")
    def update_Angle(self,angle,r_l):
        try :
            if r_l == "r" :
                self.mainRef.child("motion_analysis").child("right_angle").set(angle)
            else :
                self.mainRef.child("motion_analysis").child("left_angle").set(angle)
        except :
            print("exception")
