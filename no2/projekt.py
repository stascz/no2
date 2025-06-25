import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import stft

def wczytaj_kolumny_z_pliku(nazwa_pliku):
    kolumny = []

    with open(nazwa_pliku, "r", encoding="utf-8") as plik:
        for linia in plik:
            # Pomija puste linie
            if not linia.strip():
                continue

            # Rozdziel liczby – dowolna ilość spacji jako separator
            wartosci = linia.strip().split()

            # Jeśli to pierwsza linia, inicjalizuj kolumny
            if not kolumny:
                kolumny = [[] for _ in range(len(wartosci))]

            # Dodaj wartości do odpowiednich kolumn
            for i, wartosc in enumerate(wartosci):
                kolumny[i].append(float(wartosc))  # konwertujemy do float

    return kolumny

# Przykład użycia
nazwa_pliku = "dane.txt"
kolumny = wczytaj_kolumny_z_pliku(nazwa_pliku)

czas=np.array(kolumny[0])
hplus=np.array(kolumny[1])
hcross=np.array(kolumny[2])
RealA22=np.array(kolumny[3])
ImA22=np.array(kolumny[4])
RealA21=np.array(kolumny[5])
ImA21=np.array(kolumny[6])
RealA20=np.array(kolumny[7])
ImA20=np.array(kolumny[8])
RealA2minus1=np.array(kolumny[9])
ImA2minus1=np.array(kolumny[10])
RealA2minus2=np.array(kolumny[11])
ImA2minus2=np.array(kolumny[12])
dEdt=np.array(kolumny[13])
Etot=np.array(kolumny[14])

# Tworzymy tablicę z liczbami 1, 2, 3 ... o tym samym rozmiarze
nat = np.arange(1, len(RealA22) + 1)








y22=np.sqrt(5/(64*np.pi))
y21=np.sqrt(5/(16*np.pi))
y20=np.sqrt(15/(32*np.pi))
y2minus1=np.sqrt(5/(16*np.pi))
y2minus2=np.sqrt(5/(64*np.pi))

hplus1=y22*RealA22+y21*RealA21+y20*RealA20+y2minus1*RealA2minus1+y2minus2*RealA2minus2
hcross1=-y22*ImA22-y21*ImA21-y20*ImA20-y2minus1*ImA2minus1-y2minus2*ImA2minus2

print(hplus)
print(hcross)
print(hplus1)
print(hcross1)

# Dopasowanie prostej: y = a*x + b
aplus1, bplus1 = np.polyfit(hplus, hplus1, 1)
across1, bcross1 = np.polyfit(hcross, hcross1, 1)

print( aplus1)
print( across1)
print(bplus1)
print(bcross1)


I11 = RealA22/2 - RealA20/3
I22 = RealA22/2 + RealA20/3
I33 = 2*RealA20/3
I13 = -RealA21
I23 = ImA21
I12 = -ImA22/2

Qthetaphi = -I23
Qphiphi = I22
Qthetatheta = I33

hplus2 = Qthetatheta - Qphiphi
hcross2 = 2*Qthetaphi

hplus3 = RealA20 - RealA22
hcross3 = 2*Qthetaphi

aplus2, bplus2 = np.polyfit(hplus, hplus2, 1)
across2, bcross2 = np.polyfit(hcross, hcross2, 1)

aplus3, bplus3 = np.polyfit(hplus, hplus3, 1)
across3, bcross3 = np.polyfit(hcross, hcross3, 1)

print(aplus2)
print(across2)
print(bplus2)
print(bcross2)

print(aplus3)
print(across3)
print(bplus3)
print(bcross3)

# Funkcja do jednego uśrednienia
def average_once(signal, window=3):
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode='same')

# Wielokrotne uśrednianie
dEdt7 = dEdt
n_iterations = 7
for _ in range(n_iterations):
    dEdt7 = average_once(dEdt7, len(dEdt))

# FFT
A22 = RealA22 + 1j*ImA22
A21 = RealA21 + 1j*ImA21
A20 = RealA20 + 1j*ImA20
A2minus1 = RealA2minus1 + 1j*ImA2minus1
A2minus2 = RealA2minus2 + 1j*ImA2minus2

f22, t22, ftA22 = stft(A22, fs=5000, nperseg=75)
f21, t21, ftA21 = stft(A21, fs=5000, nperseg=75)
f20, t20, ftA20 = stft(A20, fs=5000, nperseg=75)
f2minus1, t2minus1, ftA2minus1 = stft(A2minus1, fs=5000, nperseg=75)
f2minus2, t2minus2, ftA2minus2 = stft(A2minus2, fs=5000, nperseg=75)





plt.figure()
plt.plot(czas, hplus, marker='.', linestyle='', color="b", markersize=2, label="hplus")
plt.plot(czas, hplus1/aplus1, marker='.', linestyle='', color="r", markersize=2, label="hplus1/aplus1, aplus1=1.414213562532445")
plt.title("hplus z harmonijek ")
plt.legend()

plt.figure()
plt.plot(czas, hcross, marker='.', linestyle='', color="b", markersize=2, label="hcross")
plt.plot(czas, hcross1/across1, marker='.', linestyle='', color="r", markersize=2, label="hcross1/across1, across1=1.4142135626294037")
plt.title("hcross z harmonijek ")
plt.legend()

plt.figure()
plt.plot(czas, hplus, marker='.', linestyle='', color="b", markersize=2, label="hplus")
plt.plot(czas, hplus2/aplus2, marker='.', linestyle='', color="r", markersize=2, label="hplus2/aplus2, aplus2=0.7567292939023331")
plt.title("hplus z momentu kwadrupolowego z założeniem TT")
plt.legend()

plt.figure()
plt.plot(czas, hcross, marker='.', linestyle='', color="b", markersize=2, label="hcross")
plt.plot(czas, hcross2/across2, marker='.', linestyle='', color="r", markersize=2, label="hcross2/across2, across2=4.483992973931013")
plt.title("hcross z momentu kwadrupolowego z założeniem TT")
plt.legend()

plt.figure()
plt.plot(czas, hplus, marker='.', linestyle='', color="b", markersize=2, label="hplus")
plt.plot(czas, hplus3/aplus3, marker='.', linestyle='', color="r", markersize=2, label="hplus3/aplus3, aplus3=2.570420798434107")
plt.title("hplus z momentu kwadrupolowego bez założenia TT")
plt.legend()

plt.figure()
plt.plot(czas, hcross, marker='.', linestyle='', color="b", markersize=2, label="hcross")
plt.plot(czas, hcross3/across3, marker='.', linestyle='', color="r", markersize=2, label="hcross3/across3, across3=4.483992973931013")
plt.title("hcross z momentu kwadrupolowego bez założenia TT")
plt.legend()

plt.figure()
plt.plot(nat, abs(hplus-hplus1/aplus1), marker='.', linestyle='', color="b", markersize=2)

plt.figure()
plt.plot(nat, abs(hcross-hcross1/across1), marker='.', linestyle='', color="b", markersize=2)

plt.figure()
plt.plot(nat, abs(hplus-hplus2/aplus2), marker='.', linestyle='', color="b", markersize=2)

plt.figure()
plt.plot(nat, abs(hcross-hcross2/across2), marker='.', linestyle='', color="b", markersize=2)

plt.figure()
plt.plot(nat, abs(hplus-hplus3/aplus3), marker='.', linestyle='', color="b", markersize=2)

plt.figure()
plt.plot(nat, abs(hcross-hcross3/across3), marker='.', linestyle='', color="b", markersize=2)

plt.figure()
plt.plot(czas,dEdt , marker='.', linestyle='', color="b", markersize=2, label="")
#plt.plot(czas,dEdt7 , marker='.', linestyle='', color="b", markersize=2)
plt.yscale("log")
plt.xlabel("czas po odbiciu (ms)")
plt.ylabel("jasność fali graw. (10^42 erg/s)")

plt.figure()
plt.plot(czas,Etot , marker='.', linestyle='', color="b", markersize=2, label="")
plt.yscale("log")
plt.xlabel("czas po odbiciu (ms)")
plt.ylabel("energia fali graw. (10^45 erg)")









#plt.figure()
#plt.plot(nat, abs(ImA22+ImA2minus2), marker='.', linestyle='', color="b", markersize=2)




plt.show()