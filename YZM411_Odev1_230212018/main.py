import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# YARDIMCI FONKSİYONLAR VE GÖRÜNTÜ YÜKLEME
# ==============================================================================

def ortami_ve_parametreleri_yazdir(dosyalar):
    """Adım 8f: Paket sürümleri, giriş dosyaları ve çalıştırma sırasını yazdırır."""
    print("=" * 75)
    print("YZM 411 BİLGİSAYARLI GÖRÜ - ÖDEV #1 ÇALIŞTIRMA KAYDI")
    print("=" * 75)
    print(f"Python Sürümü     : {sys.version.split()[0]}")
    print(f"NumPy Sürümü      : {np.__version__}")
    print(f"OpenCV Sürümü     : {cv2.__version__}")
    print(f"Giriş Dosyaları   : {dosyalar}")
    print("Çalıştırma Sırası : 1) Veri Okuma -> 2) H1 Sayısal Yapı -> 3) H2 Renk Eşikleme "
          "-> 4) H3 Parlaklık/Gama -> 5) H4 Küçültme/Örnekleme")
    print("=" * 75 + "\n")


def goruntuleri_yukle(dosya_sozlugu):
    """OpenCV ile BGR formatında okur, RGB ve HSV karşılıklarını hazırlar."""
    veri = {}
    for ad, yol in dosya_sozlugu.items():
        if not os.path.exists(yol):
            raise FileNotFoundError(f"Görüntü bulunamadı: {yol}. Lütfen dosya adını kontrol edin.")
        bgr = cv2.imread(yol)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        veri[ad] = {"bgr": bgr, "rgb": rgb, "hsv": hsv, "yol": yol}
    return veri


# ==============================================================================
# ADIM 4 (HEDEF 1): GÖRÜNTÜNÜN SAYISAL YAPISI VE PİKSEL ERİŞİMİ
# ==============================================================================

def hedef1_sayisal_yapi(veri):
    print(">>> ADIM 4 (HEDEF 1): GÖRÜNTÜNÜN SAYISAL YAPISI İNCELENİYOR...")
    normal_bgr = veri["Normal"]["bgr"]
    normal_rgb = veri["Normal"]["rgb"]
    H, W, C = normal_rgb.shape

    print(f"Boyut (H x W x C) : {normal_rgb.shape} -> Yükseklik={H}, Genişlik={W}, Kanal={C}")
    print(f"Veri Türü (dtype) : {normal_rgb.dtype}")
    print(f"Değer Aralığı     : min={normal_rgb.min()}, max={normal_rgb.max()} (0-255 aralığı)\n")

    # Sahnedeki 3 farklı fiziksel bölgeden sabit oransal koordinatlar (x, y)
    # 1) Rubik küpünün üst mavi yüzeyi (orta-üst mavi kare)
    # 2) Beyaz A4 kağıdının orta bölgesi
    # 3) Sağdaki ilaç prospektüsü (ANDOREX) metin/kağıt bölgesi
    noktalar = {
        "Rubik Küpü (Mavi Yüzey)": (int(W * 0.165), int(H * 0.455)),
        "Beyaz A4 Kağıt (Merkez)": (int(W * 0.400), int(H * 0.550)),
        "Prospektüs (Yazılı Bölge)": (int(W * 0.650), int(H * 0.560)),
    }

    # OpenCV BGR vs RGB erişim doğrulaması: img[y, x, kanal]
    ornek_x, ornek_y = noktalar["Rubik Küpü (Mavi Yüzey)"]
    b_val, g_val, r_val = (
        normal_bgr[ornek_y, ornek_x, 0],
        normal_bgr[ornek_y, ornek_x, 1],
        normal_bgr[ornek_y, ornek_x, 2],
    )
    print(f"NumPy [y, x, kanal] ve BGR->RGB Doğrulaması (Piksel x={ornek_x}, y={ornek_y}):")
    print(f"  bgr_img[{ornek_y}, {ornek_x}] -> B={b_val}, G={g_val}, R={r_val}")
    print(f"  rgb_img[{ornek_y}, {ornek_x}] -> R={normal_rgb[ornek_y, ornek_x, 0]}, "
          f"G={normal_rgb[ornek_y, ornek_x, 1]}, B={normal_rgb[ornek_y, ornek_x, 2]}\n")

    print(f"{'Bölge Adı':<28} | {'Koordinat (x, y)':<18} | {'Normal RGB':<17} | {'Gölge RGB':<17} | {'Lamba RGB':<17}")
    print("-" * 106)
    for bolge_adi, (x, y) in noktalar.items():
        rgb_n = tuple(int(v) for v in veri["Normal"]["rgb"][y, x])
        rgb_g = tuple(int(v) for v in veri["Gölge"]["rgb"][y, x])
        rgb_l = tuple(int(v) for v in veri["Lamba"]["rgb"][y, x])
        print(f"{bolge_adi:<28} | (x={x:4d}, y={y:4d})  | {str(rgb_n):<17} | {str(rgb_g):<17} | {str(rgb_l):<17}")
    print("\n")

    # Görselleştirme: Seçilen 3 pikseli Normal görüntü üzerinde işaretle
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (isik_adi, icerik) in enumerate(veri.items()):
        axes[idx].imshow(icerik["rgb"])
        axes[idx].set_title(f"{isik_adi} Işık ({icerik['yol']})")
        for bolge_adi, (x, y) in noktalar.items():
            axes[idx].plot(x, y, marker="o", markersize=8, markeredgecolor="yellow", markerfacecolor="red")
            axes[idx].text(x + int(W * 0.015), y, bolge_adi.split()[0], color="yellow",
                           fontsize=9, weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
        axes[idx].axis("off")
    plt.suptitle("Adım 4 (H1): Üç Fotoğrafta İncelenen Sabit Piksel Konumları", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig("hedef1_pikseller.png", dpi=200)
    plt.close()


# ==============================================================================
# ADIM 5 (HEDEF 2): RENK EŞİKLERİNİ SABİTLEYİP KARŞILAŞTIRMA
# ==============================================================================

def hedef2_renk_esikleme(veri):
    print(">>> ADIM 5 (HEDEF 2): SABİT RGB VE HSV EŞİKLERİ İLE MASKELEME...")
    normal_rgb = veri["Normal"]["rgb"]
    normal_hsv = veri["Normal"]["hsv"]
    H, W, _ = normal_rgb.shape
    toplam_piksel = H * W

    # Normal ışıklı fotoğrafta Rubik küpünün mavi yüzeyinden küçük bir ROI (bölge) seçimi
    y1, y2 = int(H * 0.43), int(H * 0.49)
    x1, x2 = int(W * 0.145), int(W * 0.185)
    roi_rgb = normal_rgb[y1:y2, x1:x2]
    roi_hsv = normal_hsv[y1:y2, x1:x2]

    print(f"Referans ROI (Rubik Küpü Mavi Yüzey): x=[{x1}:{x2}], y=[{y1}:{y2}]")
    print(f"  ROI Ortalama RGB : {roi_rgb.mean(axis=(0,1)).round(1)}")
    print(f"  ROI Ortalama HSV : {roi_hsv.mean(axis=(0,1)).round(1)}")

    # Normal ışıktaki mavi ROI dağılımına göre dondurulan (sabitlenen) eşikler
    # Mavi nesne: B kanalı yüksek, G kanalı orta-yüksek (camgöbeği/mavi), R kanalı düşük
    r_max = int(np.percentile(roi_rgb[:, :, 0], 95) + 15)
    g_min = int(max(0, np.percentile(roi_rgb[:, :, 1], 5) - 15))
    b_min = int(max(0, np.percentile(roi_rgb[:, :, 2], 5) - 15))

    # OpenCV 8-bit HSV: H [0-179], S [0-255], V [0-255]
    h_min = int(max(85, np.percentile(roi_hsv[:, :, 0], 2) - 8))
    h_max = int(min(130, np.percentile(roi_hsv[:, :, 0], 98) + 8))
    s_min = int(max(60, np.percentile(roi_hsv[:, :, 1], 5) - 20))
    v_min = int(max(60, np.percentile(roi_hsv[:, :, 2], 5) - 20))

    print("\nSabitlenen (Dondurulan) Eşikler:")
    print(f"  RGB Kuralı : R < {r_max}, G > {g_min}, B > {b_min}")
    print(f"  HSV Kuralı : H in [{h_min}, {h_max}], S > {s_min}, V > {v_min}\n")

    print(f"{'Aydınlatma':<12} | {'RGB Seçilen Piksel':<20} | {'RGB Alan Oranı (%)':<20} | "
          f"{'HSV Seçilen Piksel':<20} | {'HSV Alan Oranı (%)':<20}")
    print("-" * 102)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for satir, (isik_adi, icerik) in enumerate(veri.items()):
        rgb = icerik["rgb"]
        hsv = icerik["hsv"]

        # Sabit RGB Maskesi
        rgb_mask = (rgb[:, :, 0] < r_max) & (rgb[:, :, 1] > g_min) & (rgb[:, :, 2] > b_min)
        # Sabit HSV Maskesi
        hsv_mask = (
            (hsv[:, :, 0] >= h_min) & (hsv[:, :, 0] <= h_max) &
            (hsv[:, :, 1] > s_min) & (hsv[:, :, 2] > v_min)
        )

        rgb_sayi = int(np.sum(rgb_mask))
        hsv_sayi = int(np.sum(hsv_mask))
        rgb_oran = (rgb_sayi / toplam_piksel) * 100.0
        hsv_oran = (hsv_sayi / toplam_piksel) * 100.0

        print(f"{isik_adi:<12} | {rgb_sayi:<20d} | %{rgb_oran:<19.3f} | {hsv_sayi:<20d} | %{hsv_oran:<19.3f}")

        # Görselleştirme: Özgün Görüntü, RGB Maskesi, HSV Maskesi
        axes[satir, 0].imshow(rgb)
        axes[satir, 0].set_title(f"{isik_adi} - Özgün Fotoğraf")
        axes[satir, 0].axis("off")

        axes[satir, 1].imshow(rgb_mask, cmap="gray")
        axes[satir, 1].set_title(f"{isik_adi} - RGB Maskesi (Alan: %{rgb_oran:.2f})")
        axes[satir, 1].axis("off")

        axes[satir, 2].imshow(hsv_mask, cmap="gray")
        axes[satir, 2].set_title(f"{isik_adi} - HSV Maskesi (Alan: %{hsv_oran:.2f})")
        axes[satir, 2].axis("off")

    print("\n")
    plt.suptitle("Adım 5 (H2): Sabit RGB ve HSV Eşiklerinin Üç Aydınlatmadaki 6 Maskesi", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig("hedef2_maskeler.png", dpi=200)
    plt.close()


# ==============================================================================
# ADIM 6 (HEDEF 3): PARLAKLIK KAYDIRMASI VE GAMA DÖNÜŞÜMLERİ
# ==============================================================================

def parlaklik_kaydir(img_rgb, b=40.0):
    """J = clip(I + b, 0, 255) işlemini float üzerinde uygular."""
    img_float = img_rgb.astype(np.float32)
    ham_sonuc = img_float + float(b)
    kirpilan_255_sayisi = int(np.sum(ham_sonuc > 255.0))
    kirpilan_0_sayisi = int(np.sum(ham_sonuc < 0.0))
    sonuc = np.clip(ham_sonuc, 0.0, 255.0).astype(np.uint8)
    return sonuc, kirpilan_255_sayisi, kirpilan_0_sayisi


def gama_donusumu(img_rgb, gamma=0.5):
    """J = 255 * (I / 255)^gamma işlemini float üzerinde uygular."""
    img_float = img_rgb.astype(np.float32)
    donusum = 255.0 * np.power(img_float / 255.0, float(gamma))
    sonuc = np.clip(donusum, 0.0, 255.0).astype(np.uint8)
    return sonuc


def hedef3_parlaklik_ve_gama(veri):
    print(">>> ADIM 6 (HEDEF 3): PARLAKLIK VE GAMA DÖNÜŞÜMLERİ UYGULANIYOR...")
    normal_rgb = veri["Normal"]["rgb"]
    ort_ozgun = float(normal_rgb.mean())

    img_b40, kirpilan_ust, _ = parlaklik_kaydir(normal_rgb, b=40.0)
    img_g05 = gama_donusumu(normal_rgb, gamma=0.5)
    img_g20 = gama_donusumu(normal_rgb, gamma=2.0)

    ort_b40 = float(img_b40.mean())
    ort_g05 = float(img_g05.mean())
    ort_g20 = float(img_g20.mean())
    toplam_kanal_elemani = normal_rgb.size

    print(f"Özgün Görüntü Ortalaması                       : {ort_ozgun:.2f}")
    print(f"1) Parlaklık Kaydırması (b = +40):")
    print(f"   - Kırpılmasız Beklenen Teorik Ortalama      : {ort_ozgun + 40.0:.2f}")
    print(f"   - Ölçülen Gerçek Ortalama                   : {ort_b40:.2f}")
    print(f"   - 255'te Kırpılan Kanal-Piksel Sayısı       : {kirpilan_ust} / {toplam_kanal_elemani} "
          f"(%{100.0 * kirpilan_ust / toplam_kanal_elemani:.2f})")
    print(f"2) Gama Dönüşümü (gamma = 0.5) Ölçülen Ortalama : {ort_g05:.2f}")
    print(f"3) Gama Dönüşümü (gamma = 2.0) Ölçülen Ortalama : {ort_g20:.2f}\n")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    goruntuler = [
        (normal_rgb, f"Özgün (Normal)\nOrtalama: {ort_ozgun:.2f}"),
        (img_b40, f"Parlaklık (b=+40)\nÖlçülen Ort: {ort_b40:.2f} (Beklenen: {ort_ozgun+40:.1f})"),
        (img_g05, f"Gama (γ = 0.5)\nÖlçülen Ort: {ort_g05:.2f}"),
        (img_g20, f"Gama (γ = 2.0)\nÖlçülen Ort: {ort_g20:.2f}"),
    ]
    for ax, (g, baslik) in zip(axes, goruntuler):
        ax.imshow(g)
        ax.set_title(baslik, fontsize=10)
        ax.axis("off")

    plt.suptitle("Adım 6 (H3): Parlaklık Kaydırması ve Gama Dönüşümleri Karşılaştırması", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig("hedef3_parlaklik_gama.png", dpi=200)
    plt.close()


# ==============================================================================
# ADIM 7 (HEDEF 4): KÜÇÜLTMENİN VE ÖRNEKLEMENİN ETKİSİ
# ==============================================================================

def hedef4_kucultme_etkisi(veri):
    print(">>> ADIM 7 (HEDEF 4): 1/4 KÜÇÜLTME VE EN YAKIN KOMŞU İLE GERİ BÜYÜTME...")
    normal_rgb = veri["Normal"]["rgb"]
    H, W, _ = normal_rgb.shape

    # Sağdaki ilaç prospektüsü (ANDOREX) üzerindeki küçük puntolu ve başlık içeren bölgenin kesiti
    y1, y2 = int(H * 0.52), int(H * 0.68)
    x1, x2 = int(W * 0.58), int(W * 0.74)
    kesit_ozgun = normal_rgb[y1:y2, x1:x2]
    kh, kw, _ = kesit_ozgun.shape

    # Genişlik ve yükseklikte 1/4 oranında küçültme
    kucuk_w, kucuk_h = max(1, kw // 4), max(1, kh // 4)
    kucuk_nearest = cv2.resize(kesit_ozgun, (kucuk_w, kucuk_h), interpolation=cv2.INTER_NEAREST)
    kucuk_area = cv2.resize(kesit_ozgun, (kucuk_w, kucuk_h), interpolation=cv2.INTER_AREA)

    # Yalnızca göstermek için en yakın komşu (cv2.INTER_NEAREST) ile özgün boyuta geri büyütme
    buyuk_nearest = cv2.resize(kucuk_nearest, (kw, kh), interpolation=cv2.INTER_NEAREST)
    buyuk_area = cv2.resize(kucuk_area, (kw, kh), interpolation=cv2.INTER_NEAREST)

    # Bilgi kaybını sayısal olarak da gösterebilmek için MSE / PSNR ve standart sapma hesabı
    mse_nearest = float(np.mean((kesit_ozgun.astype(np.float32) - buyuk_nearest.astype(np.float32)) ** 2))
    mse_area = float(np.mean((kesit_ozgun.astype(np.float32) - buyuk_area.astype(np.float32)) ** 2))

    print(f"Yazılı Prospektüs Kesit Boyutu (Özgün) : {kh} x {kw} piksel")
    print(f"1/4 Küçültülmüş Boyut                  : {kucuk_h} x {kucuk_w} piksel")
    print(f"Geri Büyütülmüş Boyut                  : {kh} x {kw} piksel")
    print(f"Özgün Kesit ile INTER_NEAREST MSE Farkı: {mse_nearest:.2f} (Yüksek keskinlik kopması / sahte desen)")
    print(f"Özgün Kesit ile INTER_AREA MSE Farkı   : {mse_area:.2f} (Daha düşük hata, harfler grileşip bulanıklaştı)\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    axes[0].imshow(kesit_ozgun)
    axes[0].set_title(f"1) Özgün Yazılı Kesit\n({kh}x{kw} piksel)")
    axes[0].axis("off")

    axes[1].imshow(buyuk_nearest)
    axes[1].set_title(f"2) 1/4 INTER_NEAREST Küçültme\n+ INTER_NEAREST Büyütme (MSE: {mse_nearest:.1f})")
    axes[1].axis("off")

    axes[2].imshow(buyuk_area)
    axes[2].set_title(f"3) 1/4 INTER_AREA Küçültme\n+ INTER_NEAREST Büyütme (MSE: {mse_area:.1f})")
    axes[2].axis("off")

    plt.suptitle("Adım 7 (H4): Küçültme Yöntemlerinin Okunabilirlik ve Yapay Desen (Aliasing) Etkisi",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig("hedef4_ornekleme.png", dpi=200)
    plt.close()


# ==============================================================================
# ANA ÇALIŞTIRICI BLOK
# ==============================================================================

def main():
    dosya_sozlugu = {
        "Normal": "normal.jpg",
        "Gölge": "golge.jpg",
        "Lamba": "lamba.jpg",
    }

    ortami_ve_parametreleri_yazdir(dosya_sozlugu)
    veri = goruntuleri_yukle(dosya_sozlugu)

    hedef1_sayisal_yapi(veri)
    hedef2_renk_esikleme(veri)
    hedef3_parlaklik_ve_gama(veri)
    hedef4_kucultme_etkisi(veri)

    print("=" * 75)
    print("TÜM ADIMLAR TAMAMLANDI! Üretilen görsel dosyaları:")
    print("  1. hedef1_pikseller.png       (Adım 4 - 3 fotoğrafta piksel noktaları)")
    print("  2. hedef2_maskeler.png        (Adım 5 - 3 fotoğraf ve 6 RGB/HSV maskesi)")
    print("  3. hedef3_parlaklik_gama.png  (Adım 6 - b=+40, gamma=0.5, gamma=2.0)")
    print("  4. hedef4_ornekleme.png       (Adım 7 - Prospektüs metni 1/4 küçültme karşılaştırması)")
    print("=" * 75)


if __name__ == "__main__":
    main()