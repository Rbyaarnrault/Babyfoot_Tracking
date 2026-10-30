import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque 
cv.setUseOptimized(True)

def creer_kalman(x, y):
    kf = cv.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1],
                                    [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 10
    kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
    return kf

def passe_avant(positions):
    idx = [i for i, p in enumerate(positions) if p is not None]
    if not idx:
        return 0, None, [], [], [], []
    debut, fin = idx[0], idx[-1]
    n = fin - debut + 1

    kf = creer_kalman(*positions[debut])
    x_pre, P_pre, x_post, P_post = [None]*n, [None]*n, [None]*n, [None]*n

    x_pre[0],  P_pre[0]  = kf.statePost.copy(), kf.errorCovPost.copy()
    x_post[0], P_post[0] = kf.statePost.copy(), kf.errorCovPost.copy()

    for k in range(1, n):
        kf.predict()
        x_pre[k], P_pre[k] = kf.statePre.copy(), kf.errorCovPre.copy()
        p = positions[debut + k]
        if p is not None:
            kf.correct(np.array([[p[0]], [p[1]]], np.float32))
        x_post[k], P_post[k] = kf.statePost.copy(), kf.errorCovPost.copy()

    F = kf.transitionMatrix.astype(np.float64)
    conv = lambda L: [a.astype(np.float64) for a in L]
    return debut, F, conv(x_pre), conv(P_pre), conv(x_post), conv(P_post)

def passe_arriere(F, x_pre, P_pre, x_post, P_post):
    n = len(x_post)
    xs, Ps = [None]*n, [None]*n
    xs[-1], Ps[-1] = x_post[-1], P_post[-1]

    for k in range(n - 2, -1, -1):
        C = P_post[k] @ F.T @ np.linalg.inv(P_pre[k + 1])
        xs[k] = x_post[k] + C @ (xs[k + 1] - x_pre[k + 1])
        Ps[k] = P_post[k] + C @ (Ps[k + 1] - P_pre[k + 1]) @ C.T
    return xs, Ps

def lisser(positions, rayon_base=10, facteur_incertitude=2.0):
    debut, F, x_pre, P_pre, x_post, P_post = passe_avant(positions)
    if F is None:
        return [None] * len(positions)
        
    xs, Ps = passe_arriere(F, x_pre, P_pre, x_post, P_post)
    resultats = [None] * len(positions)
    for k in range(len(xs)):
        x, y = xs[k][0, 0], xs[k][1, 0]
        sigma = np.sqrt(max(Ps[k][0, 0], 0.0))
        resultats[debut + k] = (x, y, rayon_base + facteur_incertitude * sigma)
    return resultats

def combler_trous(positions, rayon_base=10):
    n = len(positions)
    porter_check = 50
    sortie = [(p[0], p[1], rayon_base) if p is not None else None for p in positions]
    i = 0
    while i < n:
        if positions[i] is not None:
            i += 1
            continue

        debut_trou = i
        while i < n and positions[i] is None:
            i += 1
        fin_trou = i

        longueur = fin_trou - debut_trou
        if longueur > 10:
            continue
        if debut_trou == 0 or fin_trou == n:
            continue

        debut_fen = max(0, debut_trou - porter_check)
        fin_fen = min(n, fin_trou + porter_check)
        fenetre = positions[debut_fen:fin_fen]

        resultat = lisser(fenetre, rayon_base=rayon_base)

        for k in range(debut_trou, fin_trou):
            sortie[k] = resultat[k - debut_fen]
    return sortie


class DetectionBabyfoot:
    def __init__(self, scale_down=1.0):
        self.scale_down = scale_down
        def get_ksize(base_size):
            taille = int(base_size * scale_down)
            return max(3, taille + 1 if taille % 2 == 0 else taille)

        k_petit = get_ksize(3)
        k_moyen = get_ksize(11)
        k_gros = get_ksize(21)

        self.kernel_moyen = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_moyen, k_moyen))
        self.kernel_gros = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_gros, k_gros))
        self.kernel_petit = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_petit, k_petit))

        self.kernel_moyen_cross = cv.getStructuringElement(cv.MORPH_CROSS, (k_moyen, k_moyen))
        self.kernel_gros_cross = cv.getStructuringElement(cv.MORPH_CROSS, (k_gros, k_gros))
        self.kernel_petit_cross = cv.getStructuringElement(cv.MORPH_CROSS, (k_petit, k_petit))

        self.largeur_proj = 1220
        self.hauteur_proj = 720
        self.derniere_matrice = None
        self.coins_lisses = None 
        self.historique_trajectoire = deque(maxlen=40)
        self.historique_global_vitesses = []
        self.historique_global_frames = []

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
    
    def apply_rectangular_contour_masque(self, img):
        test_forme = self.masked_g_b(img)
        contours, _ = cv.findContours(test_forme, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(test_forme.shape[:2], dtype=np.uint8), None
            
        cnt = max(contours, key=cv.contourArea)
        approx = cv.approxPolyDP(cv.convexHull(cnt), 0.01 * cv.arcLength(cnt, True), True)
        
        true_terrain = np.zeros(test_forme.shape[:2], dtype=np.uint8)
        cv.fillPoly(true_terrain, [approx.astype(np.int32).reshape(-1, 1, 2)], 255)
        return true_terrain, approx
    
    def detecter_balle(self, frame, polygone_terrain):
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
                centre_balle = (center_x, center_y)

        return frame, centre_balle
   
    def detecter_polygone_projection(self, frame):
        test_forme = self.masked_g_b(frame)
        contours, _ = cv.findContours(test_forme, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv.contourArea)
            if cv.contourArea(cnt) > 5000 * (self.scale_down ** 2): 
                hull = cv.convexHull(cnt)
                
                for eps in np.linspace(0.01, 0.1, 20):
                    approx = cv.approxPolyDP(hull, eps * cv.arcLength(hull, True), True)
                    if len(approx) == 4:
                        return approx
                        
                rect = cv.minAreaRect(hull)
                box = cv.boxPoints(rect)
                return np.array(box, dtype="float32")
        return None

    def ordonner_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        pts = pts.reshape((4, 2))
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    
    def projeter_vue_dessus(self, frame, polygone_terrain):
        if polygone_terrain is not None and len(polygone_terrain) == 4:
            coins_source = self.ordonner_points(polygone_terrain)
            
            if self.coins_lisses is None:
                self.coins_lisses = coins_source
            else:
                self.coins_lisses = 0.8 * self.coins_lisses + 0.2 * coins_source
            
            (tl, tr, br, bl) = self.coins_lisses
            
            if np.linalg.norm(tl - tr) < np.linalg.norm(tl - bl):
                coins_source_p = np.array([bl, tl, tr, br], dtype="float32")
            else:
                coins_source_p = self.coins_lisses

            coins_destination = np.array([
                [0, 0],
                [self.largeur_proj - 1, 0],
                [self.largeur_proj - 1, self.hauteur_proj - 1],
                [0, self.hauteur_proj - 1]
            ], dtype="float32")

            self.derniere_matrice = cv.getPerspectiveTransform(coins_source_p, coins_destination)
        
        if self.derniere_matrice is not None:
            return cv.warpPerspective(frame, self.derniere_matrice, (self.largeur_proj, self.hauteur_proj))
        return None

    def maj_graphique_babyfoot_numerique(self, centre_balle_original):
        img_bgr = np.full((self.hauteur_proj, self.largeur_proj, 3), (79, 106, 45), dtype=np.uint8) 
        cv.putText(img_bgr, "Position temps reel (Kalman)", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if centre_balle_original is not None and self.derniere_matrice is not None:
            pt = np.array([[[float(centre_balle_original[0]), float(centre_balle_original[1])]]])
            pt_trans = cv.perspectiveTransform(pt, self.derniere_matrice)
            
            x_proj = int(pt_trans[0][0][0])
            y_proj = int(pt_trans[0][0][1])
            
            cv.circle(img_bgr, (x_proj, y_proj), 12, (0, 0, 255), -1) 
            cv.circle(img_bgr, (x_proj, y_proj), 12, (255, 255, 255), 2)
            
            return img_bgr, (x_proj, y_proj)

        return img_bgr, None

    def maj_statistiques_vitesse(self, centre_projete, cpt_frame):
        vitesse = np.nan
        if centre_projete is not None:
            self.historique_trajectoire.appendleft(centre_projete)
            if len(self.historique_trajectoire) >= 2:
                dx = self.historique_trajectoire[0][0] - self.historique_trajectoire[1][0]
                dy = self.historique_trajectoire[0][1] - self.historique_trajectoire[1][1]
                vitesse = math.sqrt(dx**2 + dy**2)
        else:
            self.historique_trajectoire.clear()
            
        self.historique_global_vitesses.append(vitesse)
        self.historique_global_frames.append(cpt_frame)
        return vitesse

    def afficher_graphique_vitesse(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.historique_global_frames, self.historique_global_vitesses, color='r', linestyle='-')
        plt.title("Vitesse instantanée de la balle (lissée Kalman)")
        plt.xlabel("Num frame")
        plt.ylabel("Vitesse (mm / frame)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    nom_video = "babyfoot.mp4"
    ECHELLE_VIDEO = 0.50
    Detecteur = DetectionBabyfoot(scale_down=ECHELLE_VIDEO)
    fps = 30.0
    
    # Passe 1
    print("Début Passe 1 : Extraction des trajectoires brutes...")
    vid = cv.VideoCapture(f"Data/{nom_video}")
    if not vid.isOpened():
        raise SystemExit("Erreur: Vidéo non trouvée.")
        
    lst_coord_balle = []
    while True:
        check, frame = vid.read()
        if not check: break
        frame = cv.resize(frame, None, fx=ECHELLE_VIDEO, fy=ECHELLE_VIDEO, interpolation=cv.INTER_LINEAR)
        masque_terrain, _ = Detecteur.apply_rectangular_contour_masque(frame)
        _, coord = Detecteur.detecter_balle(frame, masque_terrain)
        lst_coord_balle.append(coord)
    vid.release()
    
    # Lissage Kalman global
    lst_combled = combler_trous(lst_coord_balle)
    
    # Passe2
    print("Début Passe 2 : Lissage et Projection...")
    vid = cv.VideoCapture(f"Data/{nom_video}")
    total = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    nom_base = nom_video.split('.')[0]
    out_mod = out_proj = out_num = None
    
    cpt_frame = 0
    try:
        while True:
            check, frame = vid.read()
            if not check: break
            frame = cv.resize(frame, None, fx=ECHELLE_VIDEO, fy=ECHELLE_VIDEO, interpolation=cv.INTER_LINEAR)
            
            modified = frame.copy()

            # Extraction robuste
            masque_terrain_bitwise, approx_precis = Detecteur.apply_rectangular_contour_masque(frame)
            approx_4_coins = Detecteur.detecter_polygone_projection(frame)

            coord_lisse = None
            if cpt_frame < len(lst_combled) and lst_combled[cpt_frame] is not None:
                x, y, r = lst_combled[cpt_frame]
                coord_lisse = (x, y)
                cv.circle(modified, (int(x), int(y)), int(r), (0, 0, 255), 2)

            frame_projetee = Detecteur.projeter_vue_dessus(modified, approx_4_coins)
            if frame_projetee is None:
                frame_projetee = np.zeros((720, 1220, 3), dtype=np.uint8) 
                
            frame_numerique, centre_projete = Detecteur.maj_graphique_babyfoot_numerique(coord_lisse)
            vitesse = Detecteur.maj_statistiques_vitesse(centre_projete, cpt_frame)

            if approx_precis is not None:
                cv.drawContours(modified, [approx_precis], -1, (255, 0, 0), 3)

            # Affichage temps réél
            cv.imshow("Video Modifiee", modified)

            if frame_projetee is not None:
                frame_projetee_reduite = cv.resize(frame_projetee, (0, 0), fx=0.6, fy=0.6)
                cv.imshow("Projection Vue de Dessus", frame_projetee_reduite)
                
            if frame_numerique is not None:
                frame_numerique_reduite = cv.resize(frame_numerique, (0, 0), fx=0.6, fy=0.6)
                cv.imshow("Graphique Numerique", frame_numerique_reduite)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): cv.waitKey(0)

            # Exportation MP4 
            if modified.ndim == 2: modified = cv.cvtColor(modified, cv.COLOR_GRAY2BGR)
            modified = modified.astype(np.uint8)

            if out_mod is None:
                fourcc = cv.VideoWriter_fourcc(*"mp4v")
                out_mod = cv.VideoWriter(f"sortie_modifie_{nom_base}.mp4", fourcc, fps, (modified.shape[1], modified.shape[0]))
                out_proj = cv.VideoWriter(f"sortie_projetee_{nom_base}.mp4", fourcc, fps, (1220, 720))
                out_num = cv.VideoWriter(f"sortie_numerique_{nom_base}.mp4", fourcc, fps, (1220, 720))

            out_mod.write(modified)
            out_proj.write(frame_projetee)
            out_num.write(frame_numerique)

            cpt_frame += 1
            if cpt_frame % 50 == 0:
                print(f"{cpt_frame}/{total} frames traitées")

    finally:
        vid.release()
        if out_mod is not None: out_mod.release()
        if out_proj is not None: out_proj.release()
        if out_num is not None: out_num.release()
        cv.destroyAllWindows()
        
    Detecteur.afficher_graphique_vitesse()
    print("Traitement terminé avec succès !")
