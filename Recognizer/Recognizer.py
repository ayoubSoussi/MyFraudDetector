import pickle

import cv2
import face_recognition
import imutils


class Recognizer :
    def __init__(self,service):
        self.myService = service
        self.nb_images = 0 ;
        self.persons_recognized = dict() ;
        self.data = pickle.loads(open("Recognizer/encodings.pickle", "rb").read())
        self.currentFrame = None
        self.rgb = None

    def addImage(self, frame):
        '''
        THis method take a the frame and extract the name of the person in it, then it add it to the list
        of detected persons during the session
        :param frame:
        :return: currentFrame : and it's the new frame where the box and the name were drawn
        '''

        self.currentFrame = frame
        self.nb_images+=1 ;
        # recognize the person in image and add him to the dictionnary(or +1 if already exists)
        self.rgb = imutils.resize(self.currentFrame, width=750)
        self.getAndSaveName(self.currentFrame)
        print(self.persons_recognized)
        return self.currentFrame

    def getAndSaveName(self,frame):
        mainFace = self.getMainFace(self.rgb)
        if mainFace == None :
            return
        box, encoding = mainFace
        r = frame.shape[1] / float(self.rgb.shape[1])
        matches = face_recognition.compare_faces(self.data["encodings"],
                                                 encoding, tolerance=0.5)
        name = "Unknown"

        # vérifier si on a trouvé une similitude
        if True in matches:
            # trouver les indices de toutes les similitudes trouvées puis
            # initialiser un dictionnaire pour compter le nombre total
            # des cas de similitude pour chaque visage
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # boucler sur les indices de similitude et maintenir un
            # compteur pour chaque visage reconnu
            for i in matchedIdxs:
                name = self.data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determiner le visage reconnu avec le grand nombre de
            # votes
            name = max(counts, key=counts.get)

        # Ajouter l'étudiant détecté à l'ensemble des étudiants
        # reconnus pendant la séance
        #ajouter le nom
        if name in self.persons_recognized.keys() :
            self.persons_recognized[name]+=1
        else :
            self.persons_recognized[name] = 1 ;

        # rescale the face coordinates
        top,right,bottom,left = box ;
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        # dessiner le nom du visage prédit dans l'image
        cv2.rectangle(frame, (left, top), (right, bottom),
                     (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                     0.75, (0, 255, 0), 2)
        self.currentFrame = frame
        # save face to firebase
        self.myService.update_face(name,self.persons_recognized[name])

    def getMainFace(self,rgb):
        boxes = face_recognition.face_locations(self.rgb, model="hog")
        if len(boxes) <= 0:
            return None
        print(boxes)
        encodings = face_recognition.face_encodings(rgb, boxes)

        maxIndex = self.getMaxIndex(boxes)
        return boxes[maxIndex],encodings[maxIndex]
    def getMaxIndex(self,boxes):
        max = 0
        maxIndex = 0
        for i,box in enumerate(boxes):
            val = abs(box[0]-box[2])*abs(box[1]-box[3])
            if val > max :
                max = val
                maxIndex = i

        return maxIndex
