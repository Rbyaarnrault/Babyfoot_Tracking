import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque 

img = cv.imread("baby_foot_frame.png")
if img is not None:
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

class DetectionBabyfoot:
    def __init__(self, img=None):
        self.kernel_moyen = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
        self.kernel_gros = cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21))
        self.kernel_petit = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        self.kernel_moyen_cross = cv.getStructuringElement(cv.MORPH_CROSS, (11, 11))
        self.kernel_gros_cross = cv.getStructuringElement(cv.MORPH_CROSS, (21, 21))
        self.kernel_petit_cross = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

        self.largeur_proj = 1220
        self.hauteur_proj = 720
        self.derniere_matrice = None

        # Variables de tracking
        self.historique_trajectoire = deque(maxlen=40)
        self.current_tracking_time = 0
        self.max_tracking_time = 0
        self.historique_global_vitesses = []
        self.historique_global_frames = []

        if img is not None:
            self.mask_green = cv.inRange(hsv, (40, 60, 50), (85, 170, 255))
            self.mask_blue  = cv.inRange(hsv, (100, 150, 50), (130, 255, 255))
            print(55 / 2)
            print(21 * 255 / 100)
            print(43 * 255 / 100)


    def masked_g_b(self, param_img):
        hsv = cv.cvtColor(param_img, cv.COLOR_BGR2HSV)
        mask_green_1 = cv.inRange(hsv, (41, 52, 43), (90, 140, 220))
        lower = np.array((41, 52, 43))
        upper = np.array((90, 140, 220))
        mask = cv.inRange(hsv, lower, upper)

        largeur_bordure = 60
        border = cv.copyMakeBorder(mask_green_1, largeur_bordure, largeur_bordure, largeur_bordure, largeur_bordure, borderType=cv.BORDER_REPLICATE)    

        opened = cv.morphologyEx(border, cv.MORPH_OPEN, self.kernel_petit, iterations=3)
        opened = cv.morphologyEx(opened, cv.MORPH_CLOSE, self.kernel_moyen, iterations=3)
        opened = cv.morphologyEx(opened, cv.MORPH_OPEN, self.kernel_petit, iterations=13)
        opened = cv.morphologyEx(opened, cv.MORPH_CLOSE, self.kernel_gros, iterations=8)

        opened = opened[largeur_bordure:opened.shape[0] - largeur_bordure,
                        largeur_bordure:opened.shape[1] - largeur_bordure]
        return opened
    

    def apply_rectangular_contour(self, img):
        test_forme = self.masked_g_b(img)
        contours, _ = cv.findContours(test_forme, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv.contourArea)
        pre_hull = cv.convexHull(cnt)
        if len(pre_hull) != 0:
            approx = cv.approxPolyDP(pre_hull, 0.04 * cv.arcLength(pre_hull, True), True)
            cv.drawContours(img, [approx], -1, (255,0,0), 3)
        return img
    

    def apply_rectangular_contour_masque(self, img):
        test_forme = self.masked_g_b(img)
        contours, _ = cv.findContours(test_forme, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        
        approx = cv.approxPolyDP(cv.convexHull(cnt), 0.01 * cv.arcLength(cnt, True), True)
        
        cv.drawContours(test_forme, [approx], -1, (255,0,0), 3)
        
        true_terrain = np.zeros(test_forme.shape[:2], dtype=np.uint8)
        cv.fillPoly(true_terrain, [approx.astype(np.int32).reshape(-1, 1, 2)], 255)
        return true_terrain, approx
    

    # def display_lines(self, img):
    #     img_conto = img.copy()
    #     test_forme, _ = self.apply_rectangular_contour_masque(img_conto)
    #     edges = cv.Canny(test_forme, 150, 500)
    #     lines = cv.HoughLines(edges, 1, np.pi / 360, 45, None, 0, 0)
    #     for rho, theta in lines[:, 0]:
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv.line(img_conto, pt1, pt2, (255,0,0), 3, cv.LINE_AA)
    #     return img_conto
    

    def detecter_balle(self, frame, polygone_terrain):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        min_balle = np.array([5, 80, 100]) 
        max_balle = np.array([35, 255, 255])
        mask_balle = cv.inRange(hsv, min_balle, max_balle)

        mask_balle_dilate = cv.bitwise_and(polygone_terrain, mask_balle)

        opened = cv.morphologyEx(mask_balle_dilate, cv.MORPH_OPEN, self.kernel_petit_cross, iterations=1)
        opened = cv.morphologyEx(opened, cv.MORPH_CLOSE, self.kernel_moyen_cross, iterations=5)

        circles = cv.HoughCircles(opened, cv.HOUGH_GRADIENT, 2, 1000, param1=20, param2=10, minRadius=3, maxRadius=10)

        centre_balle = None
        if circles is not None:
            for c in circles[0,:]:
                center_x, center_y, radius = int(c[0]), int(c[1]), int(c[2])
                cv.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
                # Ajout : mémorisation du centre
                centre_balle = (center_x, center_y)

        return frame, centre_balle

   
    def detecter_polygone_projection(self, frame):
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask_green = cv.inRange(hsv, lower_green, upper_green)

        opened = cv.morphologyEx(mask_green, cv.MORPH_CLOSE, self.kernel_moyen, iterations=2)
        opened = cv.morphologyEx(opened, cv.MORPH_OPEN, self.kernel_moyen, iterations=2)

        contours, _ = cv.findContours(opened, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv.contourArea)
            if cv.contourArea(cnt) > 10000: 
                hull = cv.convexHull(cnt)
                approx = cv.approxPolyDP(hull, 0.04 * cv.arcLength(hull, True), True)
                if len(approx) == 4:
                    return approx
        return None


    def ordonner_points(self, pts):
        #Rectangle vide 
        rect = np.zeros((4, 2), dtype="float32")
        pts = pts.reshape((4, 2))
        s = pts.sum(axis=1)

        #Sommets
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    

    def projeter_vue_dessus(self, frame, polygone_terrain):
        #rectangle parfait
        if polygone_terrain is not None and len(polygone_terrain) == 4:
            coins_source = self.ordonner_points(polygone_terrain)
            (tl, tr, br, bl) = coins_source
            
            largeur_video = np.linalg.norm(tl - tr)
            hauteur_video = np.linalg.norm(tl - bl)
            
            if largeur_video < hauteur_video:
                coins_source = np.array([bl, tl, tr, br], dtype="float32")

            coins_destination = np.array([
                [0, 0],
                [self.largeur_proj - 1, 0],
                [self.largeur_proj - 1, self.hauteur_proj - 1],
                [0, self.hauteur_proj - 1]
            ], dtype="float32")

            #Projection native opencv
            self.derniere_matrice = cv.getPerspectiveTransform(coins_source, coins_destination)
        
        if self.derniere_matrice is not None:
            return cv.warpPerspective(frame, self.derniere_matrice, (self.largeur_proj, self.hauteur_proj))
        return None


    def afficher_graphique_vitesse(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.historique_global_frames, self.historique_global_vitesses, color='r', linestyle='-')
        plt.title("Vitesse instantanée de la balle (avec occlus.)")
        plt.xlabel("Num frame")
        plt.ylabel("Vitesse (mm / frame)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    img = cv.imread("baby_foot_frame.png") 
    nom_video = "babyfoot3.mp4"
    Detecteur = DetectionBabyfoot(img)
    
    vid = cv.VideoCapture(f"Data/{nom_video}")
    if not vid.isOpened():
        raise SystemExit("Echec lecture.")
    
    cpt_frame = 0

    while True:
        check, frame = vid.read()
        if not check:
            print("Fin vidéo")
            break

        original = frame.copy()

        # Masque Terrain Paul
        masque_terrain_bitwise, _ = Detecteur.apply_rectangular_contour_masque(frame)
        
        # Détection des 4 coins stable Ryan
        polygone_stable_projection = Detecteur.detecter_polygone_projection(frame)

        # Détection balle Ryan + Paul
        modified, centre_balle = Detecteur.detecter_balle(frame.copy(), masque_terrain_bitwise)

        # Projection Ryan
        frame_projetee = Detecteur.projeter_vue_dessus(modified, polygone_stable_projection)

        cv.imshow("original", original)
        cv.imshow("modifie", modified)

        if frame_projetee is not None:
            frame_projetee_reduite = cv.resize(frame_projetee, (0, 0), fx=0.6, fy=0.6)
            cv.imshow("Projection Vue de Dessus", frame_projetee_reduite)

        if cv.waitKey(30) & 0xFF == ord('q'):
            break
        elif cv.waitKey(30) & 0xFF == ord('p'):
            cv.waitKey(0)

        cpt_frame += 1

    vid.release()
    cv.destroyAllWindows()
    
    # graphique vitesse
    Detecteur.afficher_graphique_vitesse()

    if img is not None:
        plt.imshow(cv.cvtColor(Detecteur.masked_g_b(img), cv.COLOR_BGR2RGB))
        plt.show()

        plt.imshow(cv.cvtColor(Detecteur.apply_rectangular_contour(img.copy()), cv.COLOR_BGR2RGB))
        plt.show()

        masque_test, _ = Detecteur.apply_rectangular_contour_masque(img.copy())
        plt.imshow(cv.cvtColor(masque_test, cv.COLOR_BGR2RGB))
        plt.show()