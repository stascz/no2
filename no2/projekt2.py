import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.signal import stft
from scipy.fft import fft, fftfreq

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


c = 299792458
G = int(6.6743E11)

# STFT
A22 = RealA22 + 1j*ImA22
A21 = RealA21 + 1j*ImA21
A20 = RealA20 + 1j*ImA20
A2minus1 = RealA2minus1 + 1j*ImA2minus1
A2minus2 = RealA2minus2 + 1j*ImA2minus2

f22, t22, ft1A22 = stft(A22, fs=5000, nperseg=75)
f21, t21, ft1A21 = stft(A21, fs=5000, nperseg=75)
f20, t20, ft1A20 = stft(A20, fs=5000, nperseg=75)
f2minus1, t2minus1, ft1A2minus1 = stft(A2minus1, fs=5000, nperseg=75)
f2minus2, t2minus2, ft1A2minus2 = stft(A2minus2, fs=5000, nperseg=75)


#dE_df1=(c**3/(16*np.pi*G))*(2*np.pi*f22)*(abs(ft1A22)**2 + abs(ft1A21)**2 + abs(ft1A20)**2 + abs(ft1A2minus1)**2 + abs(ft1A2minus2)**2)

# FFT
ft2A22 = fft(A22)
ft2A21 = fft(A21)
ft2A20 = fft(A20)
ft2A2minus1 = fft(A2minus1)
ft2A2minus2 = fft(A2minus2)

Fs = czas[0] - czas[1]  # częstotliwość próbkowania
f = fftfreq(len(czas), d=1/Fs)

dE_df2=(c**3/(16*np.pi*G))*(2*np.pi*f)*(abs(ft2A22)**2 + abs(ft2A21)**2 + abs(ft2A20)**2 + abs(ft2A2minus1)**2 + abs(ft2A2minus2)**2)


plt.figure()
#plt.plot(czas,dE_df1, marker='.', linestyle='', color="b", markersize=2, label="")
#plt.plot(czas,dEdt , marker='.', linestyle='', color="b", markersize=2)
plt.plot(czas,dE_df2, marker='.', linestyle='', color="b", markersize=2, label="")
plt.yscale("log")
plt.xlabel("czas po odbiciu (ms)")
plt.ylabel("jasność fali graw. (10^42 erg/s)")

plt.figure()
plt.plot(czas,Etot , marker='.', linestyle='', color="b", markersize=2, label="")
plt.yscale("log")
plt.xlabel("czas po odbiciu (ms)")
plt.ylabel("energia fali graw. (10^45 erg)")


plt.show()