# Praćenje pogleda vozača

## Skup Podataka GazeCapture

### Pregled
Projekat praćenja pogleda koji razmatramo koristi skup podataka GazeCapture, razvijen na MIT-u. GazeCapture predstavlja jedan od najobimnijih skupova podataka za istraživanje i razvoj tehnologija praćenja pogleda. Svojom sveobuhvatnošću i raznolikošću, on je postao temeljna komponenta u oblasti praćenja pogleda i Computer Vision-a.

### Karakteristike Skupa Podataka
- **Raznolikost podataka**: Skup podataka sadrži snimke lica i očiju stotine učesnika različitih demografskih karakteristika, obezbeđujući širok spektar varijacija u pogledu izgleda, pozicija i okruženja. 
- **Realni uslovi**: Snimci su prikupljeni u različitim realnim uslovima, uključujući varijacije osvetljenja, pozicije glave i izraza lica, čime se omogućava razvoj robustnih sistema za praćenje pogleda.
- **Anotacije pogleda**: Svaki snimak je anotiran sa tačnim informacijama o tome gde je učesnik gledao na ekranu uređaja tokom snimanja, što omogućava precizno treniranje i evaluaciju modela praćenja pogleda.
- **Mana kompatibilnosti sa različitim uređajima**: Podaci su prikupljeni koristeći Apple telefone i tablete, čime se otežava prilagođavanje na druge uređaje.

### Značaj za Projekat
- **Obuka Modela**: Skup podataka GazeCapture se koristi za obuku modela, u ovom slučaju `ITrackerModel`, koji se nalazi u GazeCapture pytorch reprezentaciji na GitHub-u.

- **Istraživanje i Inovacije**: Skup podataka predstavlja temelj za istraživanje u oblasti računarske vizije i praćenja pogleda, podstičući razvoj novih tehnika i pristupa u ovoj brzo rastućoj oblasti.


## ,,Fixing GazeCapture" na GitHub-u

Projekat "fixing-gazecapture" od strane wilsontang06 fokusira se na ispravljanje rasnih pristrasnosti u originalnom GazeCapture modelu. Model je prvobitno dizajniran za predviđanje tačke pogleda na osnovu selfi slika, ali je pokazivao slabije performanse kod osoba obojene kože u poređenju sa belcima. Projekat je uključivao ponovno treniranje modela s većim naglaskom na podacima osoba obojene kože u trening setu. Ovim ponovnim treniranjem poboljšane su performanse modela za osobe obojene kože, mada je došlo do blagog smanjenja ukupne tačnosti. Projekat naglašava važnost jednake efikasnosti modela prema različitim rasnim grupama.

Za više informacija posetite [GitHub repozitorijum](https://github.com/wilsontang06/fixing-gazecapture).

U ovaj projekat je ubačen njihov model u folderu ,,fix" koji možemo koristiti po želji za inference.

# Korišćenje modela Praćenja Pogleda

## Pregled
Ovaj kod omogućava praćenje pogleda koristeći video snimak. Potrebno je da se pri pozivu skripte unese argument kojim se bira korišćenje web kamere ili kamere sa mobilnog telefona. 
```bash
python infer_mediapipe.py 1
```
- `0` default webcam; ona koja je ugrađena u uređaj najčešće.
- `1, 2, 3...` redni brojevi eksternih kamera.
- u slučaju korišćenja kamere mobilnog telefona potrebno je instalirati aplikaciju IP Webcam, pokrenuti server na njoj i pročitati link za pristup iz aplikacije - na primer: `http://10.10.104.56:8080/video`.

#### Biblioteke i potrebne verzije
- `numpy`
- `scipy`
- `torch`
- `imutils`
- `mediapipe`
- `opencv-python`

## Glavne komponente

- `ITrackerModel`: Prilagođeni model za praćenje pogleda koji je preuzet od GazeCapture\pytorch. 

- `SubtractMean`: Klasa transformacije koja normalizuje slike.

- `transformFace`, `transformEyeL`, `transformEyeR`: Transformacije slika za lice i oči za pripremu za model.
- `checkpoint.pth`: Možete odabrati model iz foldera anja, og i fix u zavisnosti od toga koji želite da koristite.
- `Kalman filter` : Efikasno filtrira šum iz sirovih podataka o pogledu.
- `Gaussian filter` : Za neku vrstu interpolacije tačaka merenja, radi smanjenja trzavosti i nepredvidivih pokreta u putanji pogleda.

#### Pomoćne Funkcije
- `get_face_boundaries`: Identifikuje okvir lica na slici.
- `get_roi_eyes`: Izdvaja regione od interesa (ROI) očiju na osnovu obeležja.
- `getFaceGrid`: Generiše reprezentaciju mreže pozicije lica u okviru.
- `get_all_detections`: Kombinuje detekcije lica i očiju i vraća njihove ROI-e i mrežu lica.
- `test_webcam`: Glavna funkcija za pokretanje praćenja pogleda na web kameri.
- `load_checkpoint`: Učitava sačuvanu verziju modela.

#### Tok Izvršavanja
1. Model se učitava i priprema za inferenciju.
2. Funkcija `test_webcam` pokreće petlju koja hvata okvire sa web kamere.
3. Svaki okvir se obrađuje kako bi se izvukli regioni lica i očiju, kao i mreža lica.
4. Ovi ulazi se unose u model za predviđanje smera pogleda.
5. Predviđena tačka pogleda se ispisuje na terminal.
6. Petlja se nastavlja dok se ne pritisne 'q' za izlazak.


#### Napomena
- Ovaj kod pretpostavlja postojanje datoteka kontrolnih tačaka i srednjih slika za normalizaciju lica i očiju.
- Koristi MediaPipe za detekciju obeležja lica i očiju.
- Izlaz modela predstavlja predviđeni smer pogleda.

### infer_basler_mediapipe.py
U poslednjem azuriranju koda dodata je podrska za nove BASLER kamere. Ova skripta se vodi istom logikom kao i infer_mediapipe.py, ali se poziva razlicitim argumentima.

```bash
python infer_basler_mediapipe.py webcam 1
```
1. argument = `webcam`, `basler`, `phone` 
2. argument postoji samo u slucaju koriscenja web kamere ili telefona.      
    - `0` default webcam; ona koja je ugrađena u uređaj najčešće.
    - `1, 2, 3...` redni brojevi eksternih kamera.
    - u slučaju korišćenja kamere mobilnog telefona potrebno je instalirati aplikaciju IP Webcam, pokrenuti server na njoj i pročitati link za pristup iz aplikacije - na primer: `http://10.10.104.56:8080/video`.

## Zašto MediaPipe?

MediaPipe, modeli razvijeni od strane Google, izabran je zbog svoje superiornosti u detekciji obeležja lica, posebno u poređenju sa drugim opcijama poput dlib i Haar iz OpenCV. Razlozi za izbor MediaPipe-a uključuju:

- **Poboljšana Tačnost**: MediaPipe pruža preciznije i doslednije detekcije obeležja lica u odnosu na dlib i Haar.

- **Brzina i Efikasnost**: Kao ,,lightweight" rešenje, MediaPipe radi znatno brže, što je ključno za aplikacije u realnom vremenu poput praćenja pogleda.

- **Robusnost u Različitim Uslovima**: MediaPipe pokazuje snažne performanse u različitim uslovima osvetljenja i orijentacija lica.

Ukratko, kombinacija tačnosti, brzine i efikasnosti čini MediaPipe preferiranim izborom za projekat praćenja pogleda, posebno gde je obrada u realnom vremenu od suštinskog značaja.

## Kalmanov Filter 
Kalmanov Filter je algoritam koji koristi niz merenja zapaženih tokom vremena, koji sadrže statistički šum i druge nepreciznosti, i proizvodi procene nepoznatih varijabli koje su preciznije od onih zasnovanih samo na jednom merenju.
Filter procenjuje stanje sistema ažurirajući predviđanja pomoću ponderisanog proseka, gde težine daju nesigurnost svake procene.
U ovom sistemu, filter je inicijalizovan sa matricom stanja (x), matricom kovarijance greške (P), matricom prelaza stanja (A), kovarijansom šuma procesa (Q) i identičnom matricom (I).
Filter prolazi kroz dve faze: Predviđanje i Ažuriranje. U fazi predviđanja, filter koristi model sistema za predviđanje sledećeg stanja i kovarijansi greške. Tokom faze ažuriranja, inkorporira novo merenje u ova predviđanja kako bi precizirao svoje procene.

## Gaussian Smoothing
Gausovo izglađivanje se koristi za izglađivanje podataka o pogledu u realnom vremenu, smanjujući trzavost i nepredvidive pokrete u putanji pogleda. Posebno je korisno za stvaranje prirodnijeg i manje ometajućeg korisničkog iskustva.
Ono je tehnika koja primenjuje Gausov kernel na niz podataka kako bi se dobila izglađena verzija.
Funkcioniše tako što traži prosek svake tačke sa njenim susedima, pri čemu težine opadaju kako su susedi dalji od tačke. Stepenu izglađivanja upravlja standardna devijacija (sigma) Gausove raspodele.
U našoj implementaciji, primenjujemo Gausovo izglađivanje na podskupu najnovijih merenja pogleda. Ovaj pristup osigurava da je smer pogleda izglađen bez značajnog kašnjenja, održavajući odzivnost sistema.

## Korišćenje Detekcije Pogleda u Vozilima

### Upotreba

Detekcija pogleda u vozilima je tehnologija sa različitim primenama koje su usmerene na poboljšanje sigurnosti, personalizaciju iskustva vožnje i poboljšanje funkcionalnosti vozila. Evo nekoliko opštih upotreba detekcije pogleda u vozilima:

1. **Praćenje Vozača radi Sigurnosti**: Detekcija pogleda može pratiti pažnju i budnost vozača. Prateći gde vozač gleda, sistem može utvrditi da li je vozač ometen, pospan ili ne obraća pažnju na put, i može izdati upozorenja ili preduzeti korektivne akcije da spreči nesreće.

2. **Unapređena Interakcija sa Infotainment Sistemima**: Detekcija pogleda može se koristiti za interakciju sa infotainment sistemom vozila. Na primer, sistem može osetiti gde vozač gleda i automatski prikazati relevantne informacije ili kontrole, smanjujući potrebu za fizičkom interakcijom i održavajući fokus vozača na putu.

3. **Personalizovano Iskustvo Vožnje**: Tehnologija može prilagoditi postavke na osnovu preferencija i ponašanja vozača. Na primer, ako sistem detektuje da vozač često gleda u određeni displej ili kontrolu, mogao bi prilagoditi postavke displeja ili raspored kontrola da odgovara navikama vozača.

4. **Napredne Bezbednosne Funkcije**: U sistemima za pomoć vozačima (ADAS), detekcija pogleda može biti integrisana da poboljša bezbednosne funkcije. Na primer, ako sistem detektuje da vozač ne gleda u smeru gde je detektovana opasnost (kao što je nailazeće vozilo), može izdati hitnije upozorenje ili čak preduzeti preventivne akcije kao što je usporavanje vozila.

5. **Istraživanje i Razvoj**: U polju automobilskog istraživanja, detekcija pogleda može pružiti dragocene podatke o ponašanju i ergonomiji vozača, pomažući dizajnerima da kreiraju intuitivnije i sigurnije interfejse vozila.

6. **Obuka i Simulacija**: Za obuku vozača, praćenje pogleda može se koristiti za analizu i poboljšanje fokusa i pažnje pripravnika. Može pružiti povratne informacije o tome koliko dobro pripravnik održava svest o svom okruženju.

7. **Funkcije Pristupačnosti**: Za vozače sa invaliditetom, detekcija pogleda može ponuditi alternativne metode kontrole funkcija vozila, čime se povećava pristupačnost.

8. **Unapređenje Tehnologije Autonomnih Vozila**: U poluautonomnim ili potpuno autonomnim vozilima, detekcija pogleda može pružiti ključne podatke o spremnosti vozača da preuzme kontrolu nad vozilom ako je to potrebno.

Ukupno gledano, detekcija pogleda u vozilima predstavlja značajan korak ka pametnijem, sigurnijem i personalizovanijem iskustvu vožnje.


### Izazovi i Rešenja
- **Uslovi Osvetljenja**: Osvetljenje u vozilu može biti izazovno (tamno unutra, svetlo spolja, brzine promene osvetljenja tokom vožnje,..). Rešenja uključuju, na primer, ručnu kontrolu ekspozicije da se spreče previše tamne ili svetle regije lica.
- **Pozicioniranje Kamere**: Postavljanje kamere centralno, na primer pored/iznad volana, može poboljšati tačnost.

## Literatura
[1] [Eye Tracking for Everyone](https://paperswithcode.com/paper/eye-tracking-for-everyone), Kyle Kafka et al.

[2] [Kalman Filtering in the Design of Eye-Gaze-Guided Computer Interfaces](https://www.semanticscholar.org/paper/Kalman-Filtering-in-the-Design-of-Eye-Gaze-Guided-Komogortsev-Khan/f17185f549b0ae6a3de6230ebf7ce481c4d92d73),Oleg V. Komogortsev, J. Khan

[3] [ETH-XGaze](https://paperswithcode.com/dataset/eth-xgaze),Zhang et al.

[4] [The Story in Your Eyes: An Individual-difference-aware Model for Cross-person Gaze Estimation](https://arxiv.org/abs/2106.14183), Jun Bao, Buyu Liu, Jun Yu
