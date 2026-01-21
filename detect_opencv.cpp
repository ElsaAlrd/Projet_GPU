#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Erreur: caméra non ouverte" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);

    // Soustracteur de fond
    Ptr<BackgroundSubtractor> bg = createBackgroundSubtractorMOG2();

    // Compteur
    int personnes_dans_la_salle = 0;

    // ROI porte (à ajuster)
    Rect porte_roi(160, 0, 320, 480);

    // Ligne verticale
    int line_x = porte_roi.width / 2;

    int prev_position = -1;

    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        resize(frame, frame, Size(640, 480));

        rectangle(frame, porte_roi, Scalar(255, 0, 0), 2);
        Mat porte = frame(porte_roi);

        // Soustraction de fond
        Mat fgMask;
        bg->apply(porte, fgMask);

        // Nettoyage
        threshold(fgMask, fgMask, 175, 255, THRESH_BINARY);
        //adaptiveThreshold(fgMask, fgMask, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);
        //adaptiveThreshold(fgMask, fgMask, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 3);
        imshow("Threshold", fgMask);
        morphologyEx(fgMask, fgMask, MORPH_OPEN,
                     getStructuringElement(MORPH_RECT, Size(5, 5)));
        morphologyEx(fgMask, fgMask, MORPH_DILATE,
                     getStructuringElement(MORPH_RECT, Size(5, 5)));

        // Contours
        vector<vector<Point>> contours;
        findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        int current_position = -1;

        for (const auto& c : contours)
        {
            Rect r = boundingRect(c);

            // Filtre taille (personne approximative)
            if (r.area() < 1500)
                continue;

            rectangle(porte, r, Scalar(0, 255, 0), 2);

            int center_x = r.x + r.width / 2;
            current_position = center_x;

            // Visualisation centre
            circle(porte, Point(center_x, r.y + r.height / 2),
                   4, Scalar(0, 0, 255), -1);

            break; // on ne garde que le blob principal
        }

        // Ligne virtuelle
        line(porte, Point(line_x, 0), Point(line_x, porte.rows),
             Scalar(0, 255, 255), 2);

        // Comptage
        if (prev_position != -1 && current_position != -1)
        {
            if (prev_position < line_x && current_position > line_x)
                personnes_dans_la_salle++;

            if (prev_position > line_x && current_position < line_x)
                personnes_dans_la_salle--;
        }

        prev_position = current_position;

        if (personnes_dans_la_salle < 0)
            personnes_dans_la_salle = 0;

        putText(frame,
                "Personnes dans la salle: " + to_string(personnes_dans_la_salle),
                Point(10, 30),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(0, 255, 0),
                2);

        imshow("Comptage porte", frame);
        imshow("Masque mouvement", fgMask);

        if (waitKey(20) == 27)
            break;
    }

    return 0;
}
