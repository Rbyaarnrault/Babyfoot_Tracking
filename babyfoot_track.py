import cv2 as cv
import numpy as np

def main():
    print("---Début de l'analyse---")

    cap = cv.VideoCapture("Data/babyfoot3.mp4")
    
    if not cap.isOpened():
        print("Erreur : Echec de lecture.")
        return

    print("Détection en cours...\n")
    print("Pause: p\nPlay: n'importe quelle touche\nQuitter: q\n")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("-Fin de la vidéo.")
            break

            
        fen_resized_origine = cv.resize(frame, (0,0), fx=0.75, fy=0.75)
        
        # Niveau de gris et HSV
        gray = cv.cvtColor(fen_resized_origine, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(fen_resized_origine, cv.COLOR_BGR2HSV)
        min_green = np.array([35, 40, 40])
        max_green = np.array([85, 255, 255])

        fen_res = np.zeros_like(fen_resized_origine)

        # Terrain avec masque vert
        mask_green = cv.inRange(hsv,min_green, max_green)
        medianBlur = cv.medianBlur(mask_green, 11)
        contours, hierrarchy = cv.findContours(medianBlur, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        terrain_contour = max(contours, key=cv.contourArea)
        perimetre = cv.arcLength(terrain_contour, True)
        approx_polygone = cv.approxPolyDP(terrain_contour, 0.04 * perimetre, True)

        mask_rectangle = np.zeros_like(fen_resized_origine)
        cv.drawContours(fen_res, [approx_polygone], -1, (255,255,255), -1)

        # Balle avec masque jaune-orang
        min_balle = np.array([5, 100, 100]) 
        max_balle = np.array([25, 255, 255])
        mask_balle = cv.inRange(hsv, min_balle, max_balle)

        # Noyau 5x5 pixels
        kernel = np.ones((5, 5), np.uint8)
        # dilatation grossit le blanc pour recoller les morceaux de la balle
        mask_balle_dilate = cv.dilate(mask_balle, kernel, iterations=1)

        mask_balle_blurred = cv.medianBlur(mask_balle_dilate, 3)
        circles = cv.HoughCircles(mask_balle_blurred, cv.HOUGH_GRADIENT, 1, 30, param1=20, param2=10, minRadius=3, maxRadius=10)

        if circles is not None:
            # COnversion Int
            circles = np.uint16(np.around(circles))

            for c in circles[0,:]:
                center_x, center_y, radius = int(c[0]), int(c[1]), int(c[2])

                if approx_polygone is not None:
                    est_dedans = cv.pointPolygonTest(approx_polygone, (center_x, center_y), False)

                    if est_dedans >= 0:
                        # contour
                        cv.circle(fen_res, (center_x, center_y), radius, (0, 0, 255), 2)
                
        cv.imshow("Origine", fen_resized_origine)

        cv.imshow("gray_bkurred", mask_balle_blurred)
        edges_debug = cv.Canny(mask_balle_blurred, 10, 20)
        cv.imshow("Vision interne (Bords de la balle ?)", edges_debug)

        cv.imshow("Résultat", fen_res)

              
        key =  cv.waitKey(25) & 0xFF
        if key == ord('q'): # quitter
            break  
        elif key == ord('p'): # pause
            cv.waitKey(0)


    #Libération des ressources
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()