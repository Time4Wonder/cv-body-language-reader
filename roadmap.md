**Phase 1: Basis-Setup & Objekterkennung (**  
   
 Das System soll im Videostream zuverlässig erkennen, wo Gesicht und Hände sind.  
- **Umgebung:** Richte deine Python-Umgebung ein (TensorFlow oder PyTorch, OpenCV für die Bildverarbeitung).  
- **Modell-Wahl:** Binde ein kleines, ressourcenschonendes Modell wie YOLOv8 Nano ein, damit später nichts ruckelt.  
- **Pipeline:** Schreib ein Skript, das auf die Webcam zugreift und Bounding Boxes um Gesicht und Hände zieht.  
- **Zwischenziel:** Ein laufendes Live-Webcam-Skript, das dich und deine Hände in Echtzeit einrahmt.  
**Phase 2: Mimik-Auswertung **  
   
 Hier bewerten wir isoliert die Emotionen im Gesicht.  
- **Daten:** Nutze einen reduzierten AffectNet-Datensatz, um Trainingszeit zu sparen.  
- **Modell-Anpassung:** Nimm ein vortrainiertes ResNet, wirf die letzte Klassifikationsschicht raus und ersetze sie durch eine eigene für Valenz (positiv/negativ) und Arousal (Energielevel).  
- **Verknüpfung:** Das von YOLO erkannte Gesicht wird jetzt direkt an das ResNet übergeben.  
- **Zwischenziel:** Neben deinem Gesicht im Videostream werden live die V- und A-Werte angezeigt.  
**Phase 3: Gestik-Tracking**  
   
 Zusätzlich zur Mimik werten wir die Handbewegung über die Zeit aus.  
- **Tracking:** Erfasse die Mittelpunkte der YOLO-Boxen für die Hände.  
- **Glättung:** Nutze einen Kalman-Filter oder Optical Flow, um Aussetzer abzufangen, falls YOLO die Hände kurz verliert.  
- **Features:** Berechne aus den Tracking-Daten simple Features für ein bestimmtes Zeitfenster (z.B. die durchschnittliche Handgeschwindigkeit der letzten 10 Frames).  
- **Zwischenziel:** Das System gibt live Bewegungswerte wie die Handgeschwindigkeit in Pixeln aus.  
**Phase 4: Datensammlung & Klassifikator**  
   
 Das Modell lernt nun die finalen psychologischen Kategorien.  
- **Logging:** Speichere die ermittelten Werte (Valenz, Arousal, Handgeschwindigkeit) zusammen mit Zeitstempeln in einer CSV-Datei.  
- **Eigene Daten:** Setz dich vor die Kamera und simuliere für jeweils ein paar Minuten die Zielzustände (z.B. Entspannt, Fokussiert, Gestresst). Hol dir ggf. Leute dazu, um mehr Daten zu haben.  
- **Training:** Trainiere mit diesen geloggten Daten ein klassisches Machine-Learning-Modell (z.B. Random Forest oder SVM via scikit-learn).  
- **Zwischenziel:** Ein fertiges Modell, das weiß, welche Zahlenkombination zu welchem psychologischen Zustand gehört.  
**Phase 5: Live-Demo & Optimierung**  
   
 Alles wird zur fertigen Pipeline verschmolzen und für die Präsentation optimiert.  
- **Zusammenführung:** Webcam-Bild -> YOLO -> ResNet & Hand-Tracking -> Klassifikator -> finale Vorhersage.  
- **Darstellung:** Blende die Vorhersage (z.B. "Gestresst") über OpenCV gut lesbar in das Live-Bild ein.  
- **Performance:** Falls die FPS einbrechen, optimiere das Ganze (z.B. durch Multithreading oder indem YOLO nur jeden dritten Frame analysiert).  
- **Ziel:** Eine flüssig laufende Demo für die Abschlusspräsentation.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsSfYxZo/khWsYQLPJrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA4qjBdKlX6OKAAAAAElFTkSuQmCC)  
**Genereller Tipp für die Umsetzung:** Es spart extrem viel Zeit, das Rad nicht neu zu erfinden. Nutze High-Level-Bibliotheken (wie ultralytics für YOLO oder Hugging Face transformers), statt Modelle von Grund auf neu zu programmieren. So bleibt der Fokus auf der Pipeline-Architektur und den eigentlichen Daten.  
