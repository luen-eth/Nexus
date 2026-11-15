# Nexus Desktop

Türkçe Qt 6 masaüstü arayüzü; `nexus-server` çalıştırır veya mevcut yerel sunucuya bağlanır. Python **3.11+** ve **PySide6 >=6.8,<7** gerekir. Arayüz model/ikili seçimi, CPU/CUDA, KV biçimi, context/port, model yaşam döngüsü, sağlık/metrik ve son 600 günlük satırını sunar.

## Geliştirme

```bash
PYTHONPATH=desktop python3 -m nexus_desktop
PYTHONPATH=desktop python3 -m nexus_desktop --help
PYTHONPATH=desktop python3 -m nexus_desktop --version
```

`--help` ve `--version` display veya Qt yüklemesi gerektirmez. `--server /yol/nexus-server`, `--model /yol/model.gguf`, `--port 8080` başlangıç alanlarını doldurur; otomatik işlem başlatmaz. `--connect http://127.0.0.1:8080` yalnız mevcut sunucuya bağlanır.

## Linux kurulumu

Arch Linux'te `sudo pacman -S pyside6 qt6-svg qt6-wayland` ile Qt bağımlılıklarını kurun ve Nexus Desktop'ın Arch paketini kullanın. Debian/Ubuntu `.deb` paketi dağıtım paketidir; Arch için `.deb` kurmak yerine sağlanan Arch paketini kullanın. Paketleme tarifleri depo kökündeki [`packaging/`](../packaging) altındadır. Kaynaktan çalıştırırken de aynı Python modülü kullanılır. Arch bağımlılık adı resmî [`pyside6` paketidir](https://archlinux.org/packages/extra/x86_64/pyside6/).

Paket sözleşmesi: Python modülü `/usr/share/nexus-desktop/nexus_desktop`, launcher `/usr/bin/nexus-desktop`. Sunucu otomatik olarak `/usr/lib/nexus-desktop/nexus-server`, `/usr/share/nexus-desktop/bin/nexus-server`, `/usr/lib/nexus/nexus-server` veya PATH'te aranır. Geliştirme checkout'unda `target/release/nexus-server` de bulunur. Sunucu bulunamazsa dosya seçiciyle ikiliyi gösterin. GUI, sunucu bağımlılığını otomatik indirmez.

CPU ve F32 KV destekli varsayılandır. CUDA için CUDA ile derlenmiş sunucu ve uygun NVIDIA sürücüsü gerekir; seçim başarısızsa sunucu hatası günlükte gösterilir. GGUF kaynak quantization dosyadan algılanır; runtime weight storage F32'dir. Context değerini modelin desteklediği sınırla uyumlu seçin; eşzamanlı istekler ayrı KV cache kullanır.

## Bağlantı ve işlem sahipliği

Yönetim yalnız loopback HTTP adreslerine izin verir (`127.0.0.1`, `localhost`, `::1`). Başlatılan sunucu `127.0.0.1`'e bağlanır. Uygulama yalnız kendi `QProcess`'ini kapatır; haricî sunucuda “Bağlantıyı kes” işlemi sunucuya sinyal göndermez. Pencere kapatılırken uygulamaya ait sunucuya önce terminate, gecikirse kill uygulanır; UI iş parçacığı bekletilmez.

Uygulamanın başlattığı sunucu her başlatmada rastgele API anahtarı alır; anahtar process environment ile verilir, CLI argümanlarına konmaz. Haricî bağlantı anahtarı maskeli alanda yalnız oturumda tutulur. Hiçbir API anahtarı ayar dosyasına kaydedilmez; bilinen anahtarlar günlükten silinir. Başlatılan sunucuya başka bir istemci bağlamak için “API anahtarını kopyala” düğmesini kullanabilirsiniz; anahtar sunucu yeniden başlatıldığında değişir. XDG ayarları `$XDG_CONFIG_HOME/nexus-desktop/settings.json` (varsayılan `~/.config/...`) altında atomik ve 0600 izinle saklanır.

## Testler

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop python3 -m unittest discover -s desktop/tests -v
```

Offline testler mock HTTP/child process kullanır. Gerçek sunucu smoke testi ayrı opt-in'dir ve model indirmez:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=desktop python3 desktop/tests/real_server_smoke.py \
  --server target/release/nexus-server --model /yol/model-f32.gguf \
  --screenshot /tmp/nexus-desktop.png
```

Modeli “Kaldır” ile bellekten çıkarmak sunucu işlemini kapatmaz. Sunucu bu işlemde önceki model yolunu da temizler; tekrar çalıştırmak için seçili dosyayı “Yükle” ile açın. “Yenile”, halen yüklü modelin aynı dosyadan yeniden okunması içindir.
