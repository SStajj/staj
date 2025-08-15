1. Alıştırma 1 - Değer Düzeltmesi

  # YANLIŞ (şu anki kod):
  goruntu = np.array([...])  # 0 ve 1 değerleri kullanılmış

  # DOĞRU OLMASI GEREKEN:
  goruntu = np.zeros((10, 10), dtype=np.uint8)
  # Artı işareti için 255 değeri kullanılmalı
  goruntu[5, :] = 255  # Yatay çizgi
  goruntu[:, 4] = 255  # Dikey çizgi

  2. Proje 4 - Panorama Birleştirme Sorunu

  - Gerçek panorama görüntüleri ile test edilmeli (sentetik değil)
  - dnm3.jpeg ve dnm4.jpeg yerine gerçek örtüşen fotoğraflar kullanılmalı
  - Kodda 'yuz3.png' referansı var ama yanlış dosya - düzeltilmeli
  - Feather blending kısmı test edilip düzgün çalıştığı doğrulanmalı

  3. Proje 5 - Input Kaldırılması

  # KALDIRILMASI GEREKEN:
  secim = input("Gürültü giderme yöntemi seçin...")
  cekirdek_str = input("Kernel size giriniz...")

  # YERİNE:
  def main(yontem='gaussian', kernel_size=(3,3)):
      # Parametreler fonksiyon argümanı olarak alınmalı

  🟡 KOD KALİTESİ İYİLEŞTİRMELERİ

  6. Hata Yönetimi Eklenmesi

  def goruntu_yukle(dosya_yolu: str) -> np.ndarray:
      try:
          goruntu = cv2.imread(dosya_yolu)
          if goruntu is None:
              raise ValueError(f"Görüntü yüklenemedi: {dosya_yolu}")
          return goruntu
      except Exception as e:
          print(f"Hata: {e}")
          return None

  7. İngilizce Değişken İsimleri

  # ŞU ANKİ:
  gurultusuz, maskelenmis, esitlenmis

  # OLMASI GEREKEN:
  denoised, masked, equalized

  🟢 EKLENMESİ GEREKEN DOSYALAR

  8. requirements.txt Oluşturulması

  numpy==1.24.3
  opencv-python==4.8.0
  matplotlib==3.7.1
  Pillow==10.0.0

  12. Görüntü Dosyaları Kontrolü

  - lena.jpg tüm klasörlerde tekrarlanmış - tek bir data/images/ klasöründe tutulmalı
  - Gereksiz görüntü dosyaları temizlenmeli

  🔧 PERFORMANS İYİLEŞTİRMELERİ

  13. Vektörizasyon Kullanımı

  # YAVAS (Alıştırma 6'da):
  for i in range(yukseklik):
      for j in range(genislik):
          # işlem

  # HIZLI:
  # NumPy vektörizasyon kullan
  result = np.vectorize(func)(image)

  14. Memory Optimization

  # dtype belirtilmeli
  image = np.zeros((height, width), dtype=np.uint8)  # uint8 yeterli
  # float64 yerine float32 kullan gerektiğinde
