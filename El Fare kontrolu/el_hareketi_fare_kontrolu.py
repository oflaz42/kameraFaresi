# =============================================================================
# GEREKLİ KÜTÜPHANELERİ YÜKLEMEK İÇİN (Terminal / Komut İstemcisi):
#
#   pip install opencv-python mediapipe pyautogui pystray Pillow pynput
#
# Notlar:
#   - pynput yalnızca GENEL (global) klavye kısayolları için kullanılır
#     (ör. Ctrl+Alt+Space ile anında duraklat/devam et). Kurulu değilse
#     uygulama yine tam olarak çalışır; yalnızca bu ek güvenlik kısayolu
#     devre dışı kalır, panel/tepsi üzerinden duraklatma her zaman kullanılabilir.
#   - İlk çalıştırma: El tespiti için gereken küçük yapay zeka modeli
#     (birkaç MB) Google'ın resmi sunucusundan otomatik indirilir; bunun
#     için yalnızca İLK çalıştırmada internet bağlantısı gerekir. Model
#     indirildikten sonra diskte önbelleğe alınır ve bir daha indirilmez.
#   - macOS: Sistem Ayarları > Gizlilik ve Güvenlik altından bu betiği
#     çalıştıran uygulamaya (Terminal / Python) hem "Erişilebilirlik"
#     (Accessibility) hem de "Kamera" izinlerini vermeniz gerekir; aksi
#     hâlde fare hareket etmez veya kameradan görüntü alınamaz.
#   - Linux: Sistem tepsisi (system tray) desteği, masaüstü ortamına göre
#     ek paket gerektirebilir (ör. AppIndicator tabanlı ortamlarda
#     `gir1.2-appindicator3` paketi). Ayrıca önizleme/ayarlar penceresi
#     için gereken `tkinter` bazı minimal Linux kurulumlarında ayrı bir
#     sistem paketi gerektirir: `sudo apt install python3-tk`.
#   - Windows: Ek bir kurulum gerekmez. Konsol penceresinin hiç
#     görünmemesini isterseniz dosyayı `.pyw` uzantısıyla kaydedip
#     `pythonw` ile çalıştırabilir, ya da PyInstaller ile `--noconsole`
#     seçeneğini kullanarak tek bir .exe hâline getirebilirsiniz.
# =============================================================================

"""
El Hareketiyle Fare Kontrolü (Headless, Sistem Tepsisi Uygulaması)
====================================================================

Bu betik web kamerasından görüntü alır (OpenCV), MediaPipe'ın Hand
Landmarker (Tasks API) modeliyle eldeki 21 referans noktasını (landmark)
tespit eder ve:

  * İşaret parmağı ucunun (landmark 8) hareketini fare imlecine,
  * Başparmak (landmark 4) ile işaret parmağı ucunun (landmark 8)
    birbirine değmesini ("pinch") ise sol tık olayına

dönüştürür. Ek olarak, başparmak+yüzük parmağı sağ tık, başparmak+serçe
parmak basılı tutup dikey hareket ettirme ise fare tekerleği kaydırması
üretir; sol tık aynı zamanda basılı tutulup hareket ettirilirse sürükleme
(drag) olarak işler.

Uygulama, kamera görüntüsünü (el iskeleti, aktif bölge ve üç hareket
çizgisi çizilmiş hâlde) ve canlı ayar kaydırıcılarını gösteren bir kontrol
paneli ile açılır; panel sistem tepsisine küçültülebilir ama tamamen
kapatılamaz — çıkmak için tepsi menüsündeki "Çıkış" kullanılır.

Düşük ışıkta el tespitine yardımcı olmak için kareye otomatik kontrast
iyileştirmesi (CLAHE) uygulanır; imleç titremesini azaltıp hızlı
hareketlerde gecikmeyi düşük tutmak için ise sabit pencereli ortalama
yerine hıza duyarlı bir "One Euro Filter" kullanılır.

El tespiti modeli (hand_landmarker.task) diskte bulunamazsa, uygulama
onu Google'ın resmi model deposundan otomatik olarak indirip önbelleğe
alır; bu yalnızca ilk çalıştırmada gerçekleşir.

Mimari (nesne yönelimli):
    AppConfig                   -> Ayarlanabilir tüm parametreler
    OneEuroFilter / LowPassFilter -> Hıza duyarlı yumuşatma filtresi
    CoordinateSmoother            -> Ekran koordinatları için x/y filtre çifti
    CameraManager                  -> Kamera yönetimi + otomatik yeniden bağlanma
    HandTracker                    -> MediaPipe Hand Landmarker sarmalayıcısı
    MouseController                -> PyAutoGUI tabanlı imleç/tıklama/kaydırma kontrolü
    HandGestureMouseController    -> Tüm parçaları birleştiren ana kontrolcü
    TrayApplication                 -> pystray tabanlı sistem tepsisi arayüzü
    DashboardWindow                  -> Varsayılan olarak görünen kontrol paneli

Not (iş parçacığı modeli): pystray'in simge döngüsü macOS'ta ana iş
parçacığında çalışmak zorundadır; Tkinter'in de kendi ana döngüsü ana
iş parçacığını ister. Bu çakışmayı çözmek için Tkinter ana döngüsü ana
iş parçacığında çalışır, pystray ise `run_detached()` ile kendi
arka plan iş parçacığında çalışır (pystray'in resmi olarak önerdiği
entegrasyon yöntemi budur).

Çalıştırma:
    python el_hareketi_fare_kontrolu.py

Uygulamadan tamamen çıkmak için sistem tepsisindeki simgeye sağ tıklayıp
"Çıkış" seçeneğini kullanın.
"""

import os
import sys
import time
import math
import queue
import logging
import warnings
import threading
import urllib.request
from enum import IntEnum
from collections import deque
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler

warnings.filterwarnings("ignore")

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError as exc:
    sys.stderr.write(
        "[HATA] Python'un 'tkinter' modülü bulunamadı: {0}\n"
        "Windows ve macOS'ta tkinter normalde Python ile birlikte gelir. "
        "Linux'ta genellikle ayrı bir sistem paketi olarak kurulur, ör.: "
        "'sudo apt install python3-tk'. Kurduktan sonra tekrar deneyin.\n".format(exc)
    )
    raise SystemExit(1) from exc

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import pyautogui
    import pystray
    from PIL import Image, ImageDraw, ImageTk
except ImportError as exc:
    sys.stderr.write(
        "[HATA] Gerekli bir kütüphane bulunamadı: {0}\n"
        "Lütfen dosyanın en üstünde belirtilen 'pip install' komutunu "
        "çalıştırıp tekrar deneyin.\n".format(exc)
    )
    raise SystemExit(1) from exc

try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    # pynput isteğe bağlıdır: yalnızca genel (global) klavye kısayolları için
    # kullanılır. Eksikse uygulama TAMAMEN çalışmaya devam eder; sadece
    # klavye kısayolu güvenlik katmanı devre dışı kalır (panel/tepsi
    # üzerinden duraklat/devam et her zaman kullanılabilir).
    pynput_keyboard = None
    PYNPUT_AVAILABLE = False


logger = logging.getLogger("ElFareKontrolu")


def configure_logging(log_file_path):
    """Hem dosyaya hem de (varsa) konsola yazan, dönen (rotating) bir
    günlükleyici (logger) kurar.
    """
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass

    try:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    except Exception:
        pass

    return logger


@dataclass
class AppConfig:
    """Uygulamanın tüm ayarlanabilir parametrelerini bir arada tutan
    yapılandırma sınıfı.
    """

    # Kamera ayarları
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # "Aktif bölge": çerçevenin kenarlarından bırakılan boşluk oranıdır.
    # Örneğin 0.22 değeri, çerçevenin ortadaki yaklaşık %56'lık alanının
    # tüm ekrana eşlenmesini sağlar; böylece kolu aşırı hareket ettirmeye
    # gerek kalmaz.
    zone_margin_x: float = 0.22
    zone_margin_y: float = 0.22

    # Yumuşatma: "One Euro Filter" parametreleri (basit hareketli ortalama
    # yerine). mincutoff düşükse durağan haldeyken daha pürüzsüz ama hafif
    # gecikmeli olur; beta yüksekse hızlı hareket sırasında gecikme daha da
    # azalır (gecikme/titreme dengesi hıza göre otomatik ayarlanır).
    smoothing_mincutoff: float = 1.0
    smoothing_beta: float = 0.6

    # Tüm "sıkıştırma" (pinch) hareketleri için histerezisli eşik: mesafe
    # `pinch_engage_px` altına inince hareket BAŞLAR, yalnızca tekrar
    # `pinch_release_px` üzerine çıkınca SONA ERER. İki farklı eşik
    # kullanmak (aradaki boşluk sayesinde), parmaklardaki doğal titremenin
    # sınırda ileri geri zıplayıp yanlışlıkla art arda tetiklenmesini önler.
    pinch_engage_px: float = 35.0
    pinch_release_px: float = 55.0

    # Kaydırma (scroll) hassasiyeti: parmakların dikeyde kaç piksel hareket
    # etmesinin bir "tekerlek birimi" kaydırmaya karşılık geleceği. Küçük
    # değer = daha hassas/hızlı kaydırma.
    scroll_sensitivity_px: float = 12.0

    # İki elle yakınlaştırma (zoom) hassasiyeti: iki el arasındaki mesafenin
    # kaç piksel değişmesinin bir "yakınlaştırma birimi"ne karşılık geleceği.
    zoom_sensitivity_px: float = 20.0

    # İkinci bir el görüldüğünde iki elle yakınlaştırma hareketinin
    # değerlendirilip değerlendirilmeyeceği. Kapatılırsa yalnızca tek el
    # izlenir (biraz daha az işlem yükü, daha basit davranış).
    two_hand_gestures_enabled: bool = True

    # Genel (global) klavye kısayolları (pynput biçiminde). Uygulama odakta
    # olmasa bile çalışır; acil durumda anında güvenli duraklatma sağlar.
    # pynput kurulu değilse bu kısayollar sessizce devre dışı kalır, panel/
    # tepsi menüsü üzerinden duraklatma yine de her zaman kullanılabilir.
    hotkey_toggle_pause: str = "<ctrl>+<alt>+space"
    hotkey_show_dashboard: str = "<ctrl>+<alt>+m"

    # Düşük ışıkta el tespitini iyileştirmek için kareye otomatik kontrast
    # güçlendirme (CLAHE) uygulanıp uygulanmayacağı.
    low_light_enhancement: bool = True

    # El tespiti güven eşikleri (MediaPipe Hand Landmarker)
    detection_confidence: float = 0.7
    presence_confidence: float = 0.6
    tracking_confidence: float = 0.6

    # El tespiti model dosyasının (hand_landmarker.task) diskteki konumu.
    # Dosya yoksa uygulama ilk çalıştırmada otomatik olarak indirir.
    model_path: str = field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), ".el_fare_kontrolu", "hand_landmarker.task"
        )
    )

    # Kameraya yeniden bağlanma ayarları
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 30.0
    read_failure_limit: int = 5

    # Günlük (log) dosyası konumu
    log_file_path: str = field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), "el_fare_kontrolu.log"
        )
    )


class LowPassFilter:
    """One Euro Filter'ın temel yapı taşı: basit, uyarlanabilir bir alçak
    geçiren filtre. Ayrı bir sınıf olmasının nedeni, hem konum hem de hız
    (türev) sinyalini aynı mantıkla filtrelemek için iki kez kullanılması.
    """

    def __init__(self):
        self._initialized = False
        self._value = 0.0

    def apply(self, value, alpha):
        if not self._initialized:
            self._value = value
            self._initialized = True
        else:
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    def reset(self):
        self._initialized = False
        self._value = 0.0


class OneEuroFilter:
    """Casiez, Roussel ve Vogel'in "1€ Filter" adlı, etkileşimli sistemlerde
    gürültülü giriş sinyalini yumuşatmak için geliştirdiği, hıza duyarlı
    filtre. Sinyal yavaş değiştiğinde (el durağan) daha güçlü yumuşatma,
    hızlı değiştiğinde (el hızla hareket ediyor) daha az yumuşatma (daha az
    gecikme) uygular. Bu, sabit pencereli hareketli ortalamanın "ya hep
    titrek ya hep gecikmeli" ödünleşimini ortadan kaldırır.

    Tek boyutlu bir sinyal için tasarlanmıştır; x ve y için ayrı ayrı
    örneklenir (bkz. CoordinateSmoother).
    """

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_filter = LowPassFilter()
        self._dx_filter = LowPassFilter()
        self._last_time = None
        self._last_value = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value, timestamp):
        if self._last_time is None:
            dt = 1.0 / 30.0  # ilk örnek: makul bir varsayılan kare süresi
        else:
            dt = max(timestamp - self._last_time, 1e-3)
        self._last_time = timestamp

        if self._last_value is None:
            dx = 0.0
        else:
            dx = (value - self._last_value) / dt
        self._last_value = value

        edx = self._dx_filter.apply(dx, self._alpha(self.dcutoff, dt))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self._x_filter.apply(value, self._alpha(cutoff, dt))

    def reset(self):
        self._x_filter.reset()
        self._dx_filter.reset()
        self._last_time = None
        self._last_value = None


class CoordinateSmoother:
    """Ekran koordinatlarını, her eksen için ayrı bir One Euro Filter
    kullanarak yumuşatır (bkz. OneEuroFilter). Arayüz eski basit hareketli
    ortalama sürümüyle aynıdır (update/reset), yalnızca iç uygulama
    değişmiştir; bu yüzden bu sınıfı kullanan diğer kodların değişmesi
    gerekmez.
    """

    def __init__(self, mincutoff=1.0, beta=0.6):
        self.mincutoff = mincutoff
        self.beta = beta
        self._x_filter = OneEuroFilter(mincutoff=mincutoff, beta=beta)
        self._y_filter = OneEuroFilter(mincutoff=mincutoff, beta=beta)

    def update(self, x, y):
        now = time.monotonic()
        smoothed_x = self._x_filter.filter(x, now)
        smoothed_y = self._y_filter.filter(y, now)
        return smoothed_x, smoothed_y

    def reset(self):
        self._x_filter.reset()
        self._y_filter.reset()


class GesturePriority(IntEnum):
    """Sayı küçüldükçe öncelik artar (0 = en yüksek). Acil durdurma bu
    listede yer almaz: o, hareket önceliklerinin ÜSTÜNDE, tüm hareket
    işlemeyi baştan engelleyen ayrı bir kapıdır (bkz. HandGestureMouseController.run).
    LEFT_BUTTON hem tıklamayı hem sürüklemeyi kapsar; ikisi ayrı hareketler
    değil, aynı basılı-tutma mekanizmasının kısa/uzun süreli iki görünümüdür.
    """

    LEFT_BUTTON = 0   # Sol tık / sürükleme (başparmak+orta parmak)
    RIGHT_CLICK = 1   # Sağ tık (başparmak+yüzük parmağı)
    SCROLL = 2        # Kaydırma (başparmak+serçe parmak)
    ZOOM = 3          # İki elle yakınlaştırma


@dataclass
class GestureDefinition:
    """Bir el hareketinin davranışını belirleyen, dışarıdan (panelden)
    ayarlanabilir sabit parametreler.
    """

    name: str
    priority: GesturePriority
    engage_threshold_px: float
    release_threshold_px: float
    # Eşik altına inince hareketin GERÇEKTEN tetiklenmesi için gereken
    # minimum süre (saniye). Amaç: tek karelik ölçüm gürültüsünü (parmağın
    # bir anlığına eşiğin altına "kaymasını") gerçek bir tetiklemeden
    # ayırmak. Küçük tutulmalı (~1 kare) — aksi hâlde her tetikleme
    # hissedilir şekilde gecikir.
    min_hold_seconds: float = 0.03
    # Bir tetiklemeden sonra AYNI hareketin tekrar tetiklenebilmesi için
    # gereken minimum bekleme (saniye). DİKKAT: sol tık için bunu büyük
    # tutmak hızlı çift tıklamayı ENGELLER — bu yüzden sol tık 0.0 kullanır;
    # histerezis (engage≠release) zaten sınırda titremeyi önlüyor.
    cooldown_seconds: float = 0.0


class GestureState:
    """Tek bir hareketin histerezisli eşik + minimum tutma süresi +
    cooldown uygulayan durum makinesi. `update()` her karede çağrılır ve
    o karede YENİ bir tetikleme (rising edge) mi yoksa bir bırakma
    (falling edge) mi olduğunu bildirir; ikisi de aynı karede olamaz.
    """

    def __init__(self, definition):
        self.definition = definition
        self.active = False
        self.confidence = 0.0  # 0-1: mesafe eşiği ne kadar net geçtiği (gerçek bir ML skoru DEĞİL)
        self.active_since = None  # panelin "tık mı sürükleme mi" ayrımı için
        self._pending_since = None
        self._last_release_at = None

    def update(self, distance, now):
        threshold = (
            self.definition.release_threshold_px if self.active
            else self.definition.engage_threshold_px
        )
        raw_engaged = distance < threshold
        self.confidence = (
            max(0.0, min(1.0, (threshold - distance) / threshold)) if threshold > 0 else 0.0
        )

        triggered = False
        released = False

        if not self.active:
            if raw_engaged:
                if self._pending_since is None:
                    self._pending_since = now
                held_long_enough = (
                    now - self._pending_since >= self.definition.min_hold_seconds
                )
                cooldown_ok = (
                    self._last_release_at is None
                    or now - self._last_release_at >= self.definition.cooldown_seconds
                )
                if held_long_enough and cooldown_ok:
                    self.active = True
                    triggered = True
                    self.active_since = now
                    self._pending_since = None
            else:
                self._pending_since = None
        else:
            if not raw_engaged:
                self.active = False
                released = True
                self.active_since = None
                self._last_release_at = now

        return triggered, released

    def held_seconds(self, now):
        """Hareket şu an aktifse ne kadar süredir basılı tutulduğunu döner
        (panelde "tık" ile "sürükleme" ayrımı için); aktif değilse 0 döner.
        """
        if self.active_since is None:
            return 0.0
        return max(0.0, now - self.active_since)

    def force_release(self, now):
        """El/izleme kaybedildiğinde dışarıdan çağrılır: hareket aktifse
        (ör. sürükleme ortasında) düğmenin takılı kalmaması için hemen
        bırakılmış sayılır. Çağıran taraf, gerçek bırakma eylemini
        (ör. mouse.left_up()) bu metodun True dönüşüne göre kendisi yapar.
        """
        was_active = self.active
        self.active = False
        self.active_since = None
        self._pending_since = None
        if was_active:
            self._last_release_at = now
        return was_active


class PerformanceMonitor:
    """İzleme döngüsünün GERÇEK performansını ölçer. Hiçbir değer
    uydurulmaz: her metrik `time.perf_counter()` ile ölçülen gerçek
    sürelerden veya gerçek sayaçlardan hesaplanır. Kamera/MediaPipe
    olmadan (bu betiği yazarken kullanılan test ortamında olduğu gibi)
    anlamlı sayılar üretmez — bu sınıfın değeri yalnızca GERÇEK donanımda
    çalıştırıldığında ortaya çıkar.
    """

    def __init__(self, window_size=60):
        self._window_size = window_size
        self._frame_seconds = deque(maxlen=window_size)
        self._camera_read_seconds = deque(maxlen=window_size)
        self._tracking_seconds = deque(maxlen=window_size)
        self._gesture_seconds = deque(maxlen=window_size)
        self.dropped_frames = 0
        self.hand_confidence = 0.0

    def record_dropped_frame(self):
        self.dropped_frames += 1

    def record_frame(
        self, frame_seconds, camera_read_seconds=None, tracking_seconds=None, gesture_seconds=None
    ):
        self._frame_seconds.append(frame_seconds)
        if camera_read_seconds is not None:
            self._camera_read_seconds.append(camera_read_seconds)
        if tracking_seconds is not None:
            self._tracking_seconds.append(tracking_seconds)
        if gesture_seconds is not None:
            self._gesture_seconds.append(gesture_seconds)

    @staticmethod
    def _avg_ms(values):
        return (sum(values) / len(values) * 1000.0) if values else 0.0

    def snapshot(self):
        """Panelin okuyabileceği anlık bir performans özeti döner."""
        fps = 0.0
        if len(self._frame_seconds) >= 2:
            total = sum(self._frame_seconds)
            if total > 0:
                fps = len(self._frame_seconds) / total

        return {
            "fps": fps,
            "frame_time_ms": self._avg_ms(self._frame_seconds),
            "camera_read_ms": self._avg_ms(self._camera_read_seconds),
            "tracking_latency_ms": self._avg_ms(self._tracking_seconds),
            "gesture_latency_ms": self._avg_ms(self._gesture_seconds),
            "dropped_frames": self.dropped_frames,
            "hand_confidence": self.hand_confidence,
            "sample_count": len(self._frame_seconds),
        }


class GlobalHotkeyManager:
    """pynput ile sistem GENELİNDE (uygulama odakta olmasa bile) çalışan
    klavye kısayollarını dinler. PyAutoGUI yalnızca tuş GÖNDEREBİLİR,
    genel tuş DİNLEYEMEZ — bu yüzden ayrı bir kütüphane gerekir.

    pynput kurulu değilse veya dinleyici başlatılamazsa (ör. bazı
    kısıtlı Linux ortamlarında), uygulamanın geri kalanı ETKİLENMEZ;
    yalnızca bu ek güvenlik kısayolu katmanı devre dışı kalır.
    """

    def __init__(self, bindings):
        """bindings: {'<ctrl>+<alt>+space': callable, ...} (pynput biçimi)"""
        self._bindings = bindings
        self._listener = None

    def start(self):
        if not PYNPUT_AVAILABLE:
            logger.warning(
                "pynput kurulu değil; genel klavye kısayolları devre dışı. "
                "Duraklat/devam et panel veya tepsi menüsünden kullanılabilir."
            )
            return False
        try:
            self._listener = pynput_keyboard.GlobalHotKeys(self._bindings)
            self._listener.start()
            logger.info(
                "Genel klavye kısayolları etkinleştirildi: %s",
                ", ".join(self._bindings.keys()),
            )
            return True
        except Exception:
            logger.exception(
                "Genel klavye kısayolları başlatılamadı; duraklat/devam et "
                "panel veya tepsi menüsünden kullanılabilir."
            )
            self._listener = None
            return False

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                logger.debug("Klavye kısayolu dinleyicisi durdurulurken hata oluştu.", exc_info=True)
            self._listener = None


class CameraManager:
    """Web kamerasına bağlanmayı, kare okumayı ve bağlantı koptuğunda
    otomatik olarak yeniden bağlanmayı (auto-reconnect) yönetir.
    """

    def __init__(
        self,
        camera_index=0,
        width=640,
        height=480,
        base_delay=1.0,
        max_delay=30.0,
        failure_limit=5,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.failure_limit = failure_limit

        self.capture = None
        self._fail_streak = 0
        self._reconnect_attempts = 0
        self._next_attempt_at = 0.0

        self._open()

    @property
    def is_connected(self):
        return self.capture is not None

    def _open(self):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                logger.debug(
                    "Eski kamera nesnesi kapatılırken hata oluştu.", exc_info=True
                )

        if sys.platform.startswith("win"):
            capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(self.camera_index)

        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.capture = capture
            self._fail_streak = 0
            self._reconnect_attempts = 0
            logger.info("Kamera bağlantısı kuruldu (indeks=%s).", self.camera_index)
            return True

        self.capture = None
        logger.warning("Kamera açılamadı (indeks=%s).", self.camera_index)
        return False

    def read(self):
        """Kameradan bir kare okur. Başarısız olursa None döner ve
        gerekirse otomatik yeniden bağlanmayı tetikler.
        """
        if self.capture is None:
            self._maybe_reconnect()
            return None

        try:
            ok, frame = self.capture.read()
        except Exception:
            logger.exception("Kameradan kare okunurken beklenmeyen bir hata oluştu.")
            ok, frame = False, None

        if not ok or frame is None:
            self._fail_streak += 1
            logger.warning("Kameradan görüntü alınamadı (%s. deneme).", self._fail_streak)
            if self._fail_streak >= self.failure_limit:
                try:
                    self.capture.release()
                except Exception:
                    pass
                self.capture = None
                self._maybe_reconnect()
            return None

        self._fail_streak = 0
        return frame

    def _maybe_reconnect(self):
        now = time.time()
        if now < self._next_attempt_at:
            return

        delay = min(
            self.base_delay * (2 ** min(self._reconnect_attempts, 5)), self.max_delay
        )
        self._next_attempt_at = now + delay
        self._reconnect_attempts += 1
        logger.info(
            "Kameraya yeniden bağlanılıyor (deneme %s, bekleme %.1f sn).",
            self._reconnect_attempts,
            delay,
        )
        self._open()

    def release(self):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                logger.debug("Kamera serbest bırakılırken hata oluştu.", exc_info=True)
            logger.info("Kamera kaynakları serbest bırakıldı.")


class HandTracker:
    """MediaPipe Hand Landmarker (Tasks API) sarmalayıcısı.

    Bir kare üzerinde el/parmak referans noktalarını (landmark) tespit
    eder. Gerekli model dosyası (hand_landmarker.task) diskte bulunamazsa,
    Google'ın resmi model deposundan otomatik olarak indirilip önbelleğe
    alınır (yalnızca ilk çalıştırmada internet bağlantısı gerektirir).
    """

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
    MIN_VALID_MODEL_BYTES = 100_000
    DOWNLOAD_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        model_path,
        max_num_hands=1,
        detection_confidence=0.7,
        presence_confidence=0.6,
        tracking_confidence=0.6,
    ):
        self._ensure_model(model_path)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        # process() sonrası okunabilecek, en son karedeki her el için GERÇEK
        # MediaPipe güven skorları (el sırasına karşılık gelir). Ayrı bir
        # kanal olarak tutuluyor ki process()'in dönüş imzası değişmesin
        # (mevcut çağıran kodları bozmasın).
        self.last_confidences = []

    def _ensure_model(self, model_path):
        if (
            os.path.isfile(model_path)
            and os.path.getsize(model_path) >= self.MIN_VALID_MODEL_BYTES
        ):
            return

        logger.info("El tespiti modeli indiriliyor: %s", self.MODEL_URL)
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        try:
            # urlretrieve'in aksine burada bir zaman aşımı (timeout) veriyoruz;
            # aksi hâlde ağ bağlantısı donarsa uygulama başlangıçta süresiz
            # olarak asılı kalabilirdi.
            request = urllib.request.Request(
                self.MODEL_URL, headers={"User-Agent": "el-fare-kontrolu/1.0"}
            )
            with urllib.request.urlopen(
                request, timeout=self.DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                with open(model_path, "wb") as out_file:
                    while True:
                        chunk = response.read(64 * 1024)
                        if not chunk:
                            break
                        out_file.write(chunk)
        except Exception as exc:
            if os.path.isfile(model_path):
                try:
                    os.remove(model_path)
                except OSError:
                    pass
            raise RuntimeError(
                "El tespiti modeli indirilemedi. Lütfen internet bağlantınızı "
                "kontrol edin ya da modeli elle şu adresten indirip '{0}' "
                "konumuna kaydedin: {1}".format(model_path, self.MODEL_URL)
            ) from exc

        if (
            not os.path.isfile(model_path)
            or os.path.getsize(model_path) < self.MIN_VALID_MODEL_BYTES
        ):
            raise RuntimeError(
                "İndirilen model dosyası geçersiz görünüyor. Lütfen modeli elle "
                "şu adresten indirip '{0}' konumuna kaydedin: {1}".format(
                    model_path, self.MODEL_URL
                )
            )

        logger.info("Model başarıyla indirildi: %s", model_path)

    def process(self, frame_bgr):
        """Verilen BGR kareyi işler ve tespit edilen her elin 21 referans
        noktasından oluşan bir liste döner (0, 1 veya 2 el — bkz.
        `max_num_hands`). Hiçbir el tespit edilemezse boş liste döner.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        hands = result.hand_landmarks or []

        # Handedness skoru, MediaPipe'ın bu el için GERÇEKTEN döndürdüğü
        # tek güven değeridir (el "sol mu sağ mı" sınıflandırmasının
        # güveni) -- genel tespit kalitesiyle iyi ilişkilidir, bu yüzden
        # dürüstçe "el algılama güveni" olarak sunuyoruz; uydurma bir
        # sayı DEĞİLDİR.
        confidences = []
        for handedness_for_hand in (result.handedness or []):
            confidences.append(handedness_for_hand[0].score if handedness_for_hand else 0.0)
        while len(confidences) < len(hands):
            confidences.append(0.0)
        self.last_confidences = confidences

        return hands

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            logger.debug("HandLandmarker kapatılırken hata oluştu.", exc_info=True)


class MouseController:
    """PyAutoGUI kullanarak imleç konumlandırma ve tıklama işlemlerini
    yürüten sınıf.
    """

    def __init__(self):
        # Fare programatik olarak ekranın tam köşesine (0,0 vb.) taşınırsa
        # PyAutoGUI'nin güvenlik kilidini yanlışlıkla tetiklememesi için
        # hareketleri köşelerden 1 piksel içeride tutuyoruz (bkz. move_to).
        # FAILSAFE yine de açık bırakılıyor: kullanıcı fiziksel fareyi
        # bilerek tam köşeye götürürse bu, elle acil durdurma anahtarı
        # işlevi görür (bkz. HandGestureMouseController._process_landmarks).
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0

        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(
            "Ekran çözünürlüğü algılandı: %sx%s", self.screen_width, self.screen_height
        )

    def move_to(self, x, y):
        clamped_x = int(max(1, min(x, self.screen_width - 2)))
        clamped_y = int(max(1, min(y, self.screen_height - 2)))
        pyautogui.moveTo(clamped_x, clamped_y)

    def left_down(self):
        """Sol düğmeyi basılı tutar. Hemen ardından bırakılırsa normal bir
        tıklama, basılıyken imleç hareket ettirilirse sürükleme (drag)
        olarak algılanır — işletim sistemi seviyesinde, ekstra kod
        gerekmeden.
        """
        pyautogui.mouseDown(button="left")

    def left_up(self):
        pyautogui.mouseUp(button="left")

    def right_click(self):
        pyautogui.click(button="right")

    def scroll(self, amount):
        if amount != 0:
            pyautogui.scroll(int(amount))

    def zoom(self, amount):
        """Ctrl (macOS'ta Command) tuşunu basılı tutarak kaydırma gönderir;
        bu, tarayıcılar, IDE'ler, resim görüntüleyiciler gibi birçok
        uygulamada neredeyse evrensel bir yakınlaştırma kısayoludur.
        """
        if amount == 0:
            return
        modifier = "command" if sys.platform == "darwin" else "ctrl"
        try:
            pyautogui.keyDown(modifier)
            pyautogui.scroll(int(amount))
        finally:
            pyautogui.keyUp(modifier)


class HandGestureMouseController:
    """Kamera, el tespiti, koordinat eşleme, yumuşatma ve fare kontrolünü
    bir araya getiren ana kontrolcü. Kendi iş parçacığında (thread)
    çalışacak şekilde tasarlanmıştır.

    Tek el yeterlidir; her şey tek elle çalışır. İkinci bir el görülürse
    ek olarak iki elle yakınlaştırma (zoom) hareketi de kullanılabilir.

    Ana el (imleç + tek el hareketleri; hepsi başparmakla yapılan bir
    "sıkıştırma"/pinch'e dayanır, ancak HER biri farklı bir parmakla —
    böylece işaret parmağı yalnızca imleç konumlaması için kullanılır ve
    diğer hareketler onu ASLA etkilemez):
        İşaret parmağı (8)  -> yalnızca imleç konumu
        Başparmak + Orta parmak (4+12)  -> sol tık / basılı tutup sürükleme
        Başparmak + Yüzük parmağı (4+16) -> sağ tık
        Başparmak + Serçe parmak (4+20)  -> basılı tutup dikey hareketle kaydırma

    İkincil el (yalnızca ikinci bir el görüldüğünde):
        Başparmak + İşaret parmağı sıkıştırılıp tutulur, iki el birbirinden
        uzaklaştırılıp yaklaştırılır -> yakınlaştır / uzaklaştır (zoom)

    Bir anda yalnızca TEK bir hareket etkin olabilir (öncelik sırası:
    sol > sağ > kaydırma > yakınlaştırma); bu, örneğin sürüklerken
    yanlışlıkla kaydırma veya yakınlaştırmanın da tetiklenmesini engeller.

    İki el tespit edildiğinde, MediaPipe'ın kareler arasında hangi elin
    "birinci" listelendiğini garanti etmemesi nedeniyle, ana elin kimliği
    önceki karedeki bilek (landmark 0) konumuna en yakın el baz alınarak
    kararlı biçimde takip edilir (bkz. `_assign_hands`); aksi hâlde imleç,
    iki el arasında rastgele zıplayabilirdi.
    """

    INDEX_FINGER_TIP = 8
    THUMB_TIP = 4
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20
    WRIST = 0

    LOW_LIGHT_BRIGHTNESS_THRESHOLD = 100  # 0-255 ölçeğinde L kanalı ortalaması
    PINCH_HYSTERESIS_GAP_PX = 20.0  # tutturma ve bırakma eşiği arasındaki sabit fark
    DRAG_HOLD_SECONDS = 0.15  # bu süreden uzun basılı tutulursa panelde "sürükleme" gösterilir

    def __init__(self, config, on_state_change=None):
        self.config = config
        self.on_state_change = on_state_change

        self.camera = CameraManager(
            camera_index=config.camera_index,
            width=config.frame_width,
            height=config.frame_height,
            base_delay=config.reconnect_base_delay,
            max_delay=config.reconnect_max_delay,
            failure_limit=config.read_failure_limit,
        )
        self.hand_tracker = HandTracker(
            model_path=config.model_path,
            max_num_hands=2,
            detection_confidence=config.detection_confidence,
            presence_confidence=config.presence_confidence,
            tracking_confidence=config.tracking_confidence,
        )
        self.mouse = MouseController()
        self.smoother = CoordinateSmoother(
            mincutoff=config.smoothing_mincutoff, beta=config.smoothing_beta
        )
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # Her hareket için AYRI durum makinesi (histerezis + min. tutma süresi
        # + cooldown + güven + öncelik hepsi GestureDefinition içinde açıkça
        # tanımlı — tek, paylaşılan bir eşik değil).
        self._left_gesture = GestureState(GestureDefinition(
            name="sol_tik_surukle",
            priority=GesturePriority.LEFT_BUTTON,
            engage_threshold_px=config.pinch_engage_px,
            release_threshold_px=config.pinch_release_px,
            min_hold_seconds=0.03,
            cooldown_seconds=0.0,  # ÖNEMLİ: 0 -- hızlı çift tıklamayı engellememesi için
        ))
        self._right_gesture = GestureState(GestureDefinition(
            name="sag_tik",
            priority=GesturePriority.RIGHT_CLICK,
            engage_threshold_px=config.pinch_engage_px,
            release_threshold_px=config.pinch_release_px,
            min_hold_seconds=0.03,
            cooldown_seconds=0.05,
        ))
        self._scroll_gesture = GestureState(GestureDefinition(
            name="kaydirma",
            priority=GesturePriority.SCROLL,
            engage_threshold_px=config.pinch_engage_px,
            release_threshold_px=config.pinch_release_px,
            min_hold_seconds=0.03,
            cooldown_seconds=0.0,
        ))
        self._zoom_gesture = GestureState(GestureDefinition(
            name="yakinlastirma",
            priority=GesturePriority.ZOOM,
            engage_threshold_px=config.pinch_engage_px,
            release_threshold_px=config.pinch_release_px,
            min_hold_seconds=0.03,
            cooldown_seconds=0.0,
        ))
        self._scroll_reference_y = None
        self._zoom_reference_distance = None
        self._primary_wrist_normalized = None  # ana elin kimliğini kararlı tutmak için

        self.performance = PerformanceMonitor()

        # Önizleme/ayarlar penceresi ile güvenli iş parçacığı arası iletişim için:
        self._tracker_lock = threading.Lock()
        self._preview_active = threading.Event()
        self._preview_queue = queue.Queue(maxsize=1)

    @property
    def is_paused(self):
        return self._pause_event.is_set()

    def pause(self):
        if not self._pause_event.is_set():
            self._pause_event.set()
            self._release_all_gestures()
            logger.info("Kontrol duraklatıldı.")
            self._notify_state_change()

    def resume(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.smoother.reset()
            logger.info("Kontrole devam ediliyor.")
            self._notify_state_change()

    def toggle_pause(self):
        if self.is_paused:
            self.resume()
        else:
            self.pause()

    def stop(self):
        self._stop_event.set()

    def _notify_state_change(self):
        if self.on_state_change is not None:
            try:
                self.on_state_change(self.is_paused)
            except Exception:
                logger.debug(
                    "Durum değişikliği bildirimi sırasında hata oluştu.", exc_info=True
                )

    def run(self):
        """Ana izleme döngüsü. Ayrı bir iş parçacığında çalıştırılmalıdır."""
        logger.info("İzleme döngüsü başlatıldı.")

        while not self._stop_event.is_set():
            frame_start = time.perf_counter()
            try:
                read_start = time.perf_counter()
                frame = self.camera.read()
                read_seconds = time.perf_counter() - read_start

                if frame is None:
                    self.performance.record_dropped_frame()
                    self._release_all_gestures()
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)  # Ayna görüntüsü: doğal hareket hissi için

                if self._pause_event.is_set():
                    if self._preview_active.is_set():
                        self._push_preview_frame(frame, None, None)
                    continue

                if self.config.low_light_enhancement:
                    frame = self._enhance_low_light(frame)

                tracking_start = time.perf_counter()
                with self._tracker_lock:
                    hands = self.hand_tracker.process(frame)
                tracking_seconds = time.perf_counter() - tracking_start
                confidences = self.hand_tracker.last_confidences

                if not hands:
                    self.smoother.reset()
                    self._release_all_gestures()
                    self._primary_wrist_normalized = None
                    self.performance.hand_confidence = 0.0
                    self.performance.record_frame(
                        frame_seconds=time.perf_counter() - frame_start,
                        camera_read_seconds=read_seconds,
                        tracking_seconds=tracking_seconds,
                    )
                    if self._preview_active.is_set():
                        self._push_preview_frame(frame, None, None)
                    continue

                primary, secondary, primary_confidence = self._assign_hands(hands, confidences)
                self.performance.hand_confidence = primary_confidence

                gesture_start = time.perf_counter()
                self._process_landmarks(frame, primary, secondary)
                gesture_seconds = time.perf_counter() - gesture_start

                self.performance.record_frame(
                    frame_seconds=time.perf_counter() - frame_start,
                    camera_read_seconds=read_seconds,
                    tracking_seconds=tracking_seconds,
                    gesture_seconds=gesture_seconds,
                )

            except Exception:
                logger.exception(
                    "İzleme döngüsünde beklenmeyen bir hata oluştu; döngü sürdürülüyor."
                )
                time.sleep(0.1)

        self._release_all_gestures()
        self.camera.release()
        self.hand_tracker.close()
        logger.info("İzleme döngüsü sonlandırıldı, kaynaklar serbest bırakıldı.")

    def _enhance_low_light(self, frame_bgr):
        """Karanlık ortamlarda el tespitine yardımcı olmak için, yalnızca
        kare gerçekten karanlıksa (ortalama parlaklık eşik altındaysa)
        L (parlaklık) kanalına CLAHE uygular; renkleri (a/b kanalları)
        değiştirmez. Yeterince aydınlık karelerde hiçbir şey yapmaz —
        gereksiz işlem yükünden ve zaten iyi görüntüde gürültü
        artışından kaçınmak için.
        """
        try:
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)

            if l_channel.mean() >= self.LOW_LIGHT_BRIGHTNESS_THRESHOLD:
                return frame_bgr

            l_enhanced = self._clahe.apply(l_channel)
            enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
            return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        except Exception:
            logger.debug(
                "Düşük ışık iyileştirmesi uygulanırken hata oluştu.", exc_info=True
            )
            return frame_bgr

    def _safe_mouse_action(self, action_fn, description):
        """Herhangi bir PyAutoGUI fare eylemini çalıştırır; imleç ekranın
        köşesine götürülüp güvenlik kilidi tetiklenirse bunu elle yapılan
        bir acil durdurma sinyali olarak yorumlayıp kontrolü duraklatır.
        Başarılıysa True, herhangi bir hata oluştuysa False döner.
        """
        try:
            action_fn()
            return True
        except pyautogui.FailSafeException:
            logger.warning(
                "PyAutoGUI güvenlik kilidi tetiklendi (imleç ekranın köşesine "
                "götürüldü). Bu durum, elle yapılan bir acil durdurma sinyali "
                "olarak değerlendirilip kontrol duraklatılıyor."
            )
            self.pause()
            return False
        except Exception:
            logger.exception("%s sırasında hata oluştu.", description)
            return False

    def _assign_hands(self, hands, confidences=None):
        """Tespit edilen 0-2 elden hangisinin "ana el" (imleç + tek el
        hareketleri) olacağını belirler. MediaPipe, birden fazla el
        tespit edildiğinde kareler arasında sırayı sabit tutmayı garanti
        etmediğinden, ana elin kimliği önceki karedeki bilek (landmark 0)
        konumuna en yakın el baz alınarak korunur; aksi hâlde imleç iki el
        arasında rastgele zıplayabilirdi. (ana_el, ikincil_el, ana_el_güveni)
        döner; ikincil el yoksa (None) olarak döner.
        """
        if len(hands) == 1:
            primary_index = 0
        elif self._primary_wrist_normalized is None:
            primary_index = 0
        else:
            ref_x, ref_y = self._primary_wrist_normalized

            def wrist_distance(hand):
                wrist = hand[self.WRIST]
                return math.hypot(wrist.x - ref_x, wrist.y - ref_y)

            primary_index = 0 if wrist_distance(hands[0]) <= wrist_distance(hands[1]) else 1

        secondary_index = None
        if len(hands) == 2:
            secondary_index = 1 - primary_index

        primary = hands[primary_index]
        secondary = hands[secondary_index] if secondary_index is not None else None

        primary_confidence = 0.0
        if confidences and primary_index < len(confidences):
            primary_confidence = confidences[primary_index]

        primary_wrist = primary[self.WRIST]
        self._primary_wrist_normalized = (primary_wrist.x, primary_wrist.y)
        return primary, secondary, primary_confidence

    def _process_landmarks(self, frame, landmarks, secondary_landmarks=None):
        frame_height, frame_width = frame.shape[:2]

        def to_px(landmark):
            return (landmark.x * frame_width, landmark.y * frame_height)

        index_px = to_px(landmarks[self.INDEX_FINGER_TIP])
        thumb_px = to_px(landmarks[self.THUMB_TIP])
        middle_px = to_px(landmarks[self.MIDDLE_FINGER_TIP])
        ring_px = to_px(landmarks[self.RING_FINGER_TIP])
        pinky_px = to_px(landmarks[self.PINKY_TIP])

        screen_x, screen_y = self._map_to_screen(index_px, frame_width, frame_height)
        smooth_x, smooth_y = self.smoother.update(screen_x, screen_y)

        if not self._safe_mouse_action(
            lambda: self.mouse.move_to(smooth_x, smooth_y), "İmleç hareket ettirme"
        ):
            if self._preview_active.is_set():
                self._push_preview_frame(frame, landmarks, secondary_landmarks)
            return

        self._handle_gestures(thumb_px, middle_px, ring_px, pinky_px, now=time.monotonic())

        # İki elle yakınlaştırma: yalnızca ikinci bir el varsa VE ana el
        # başka bir hareket (sol/sağ/kaydırma) yapmıyorsa değerlendirilir.
        if (
            self.config.two_hand_gestures_enabled
            and not (self._left_gesture.active or self._right_gesture.active or self._scroll_gesture.active)
        ):
            self._update_zoom(landmarks, secondary_landmarks, frame_width, frame_height, now=time.monotonic())
        elif secondary_landmarks is None and self._zoom_gesture.active:
            self._zoom_gesture.force_release(time.monotonic())
            self._zoom_reference_distance = None

        if self._preview_active.is_set():
            self._push_preview_frame(frame, landmarks, secondary_landmarks)


    def _map_to_screen(self, point_px, frame_width, frame_height):
        x_px, y_px = point_px

        zone_x_min = frame_width * self.config.zone_margin_x
        zone_x_max = frame_width * (1.0 - self.config.zone_margin_x)
        zone_y_min = frame_height * self.config.zone_margin_y
        zone_y_max = frame_height * (1.0 - self.config.zone_margin_y)

        screen_x = self._map_range(x_px, zone_x_min, zone_x_max, 0, self.mouse.screen_width)
        screen_y = self._map_range(y_px, zone_y_min, zone_y_max, 0, self.mouse.screen_height)
        return screen_x, screen_y

    @staticmethod
    def _map_range(value, in_min, in_max, out_min, out_max):
        if in_max <= in_min:
            return out_min

        clamped_value = max(in_min, min(value, in_max))
        ratio = (clamped_value - in_min) / (in_max - in_min)
        return out_min + ratio * (out_max - out_min)

    # --- Hareket (gesture) durum makinesi ---
    # Üç hareketin tümü aynı histerezisli eşik çiftini kullanır: mesafe
    # `pinch_engage_px` altına inince BAŞLAR, yalnızca tekrar
    # `pinch_release_px` üzerine çıkınca SONA ERER. Aradaki boşluk, parmak
    # titremesinin eşik sınırında ileri geri zıplayıp yanlışlıkla tekrar
    # tetiklenmesini önler.

    def _handle_gestures(self, thumb_px, middle_px, ring_px, pinky_px, now):
        left_distance = math.hypot(thumb_px[0] - middle_px[0], thumb_px[1] - middle_px[1])
        self._update_left_click(left_distance, now)
        if self._left_gesture.active:
            return  # Sürüklerken diğer hareketleri değerlendirme.

        right_distance = math.hypot(thumb_px[0] - ring_px[0], thumb_px[1] - ring_px[1])
        self._update_right_click(right_distance, now)
        if self._right_gesture.active:
            return

        scroll_distance = math.hypot(thumb_px[0] - pinky_px[0], thumb_px[1] - pinky_px[1])
        self._update_scroll(scroll_distance, pinky_px[1], now)

    def _update_left_click(self, distance, now):
        triggered, released = self._left_gesture.update(distance, now)
        if triggered:
            if self._safe_mouse_action(self.mouse.left_down, "Sol düğmeyi basma"):
                logger.info(
                    "Sol düğme basıldı (mesafe=%.1f piksel, güven=%.0f%%).",
                    distance, self._left_gesture.confidence * 100,
                )
        elif released:
            if self._safe_mouse_action(self.mouse.left_up, "Sol düğmeyi bırakma"):
                logger.info("Sol düğme bırakıldı.")

    def _update_right_click(self, distance, now):
        triggered, _released = self._right_gesture.update(distance, now)
        if triggered:
            if self._safe_mouse_action(self.mouse.right_click, "Sağ tık"):
                logger.info(
                    "Sağ tık tetiklendi (mesafe=%.1f piksel, güven=%.0f%%).",
                    distance, self._right_gesture.confidence * 100,
                )

    def _update_scroll(self, distance, current_y, now):
        triggered, released = self._scroll_gesture.update(distance, now)

        if triggered:
            self._scroll_reference_y = current_y
        elif self._scroll_gesture.active and self._scroll_reference_y is not None:
            delta_y = self._scroll_reference_y - current_y
            sensitivity = max(1.0, self.config.scroll_sensitivity_px)
            whole_units = int(delta_y / sensitivity)
            if whole_units != 0:
                if self._safe_mouse_action(
                    lambda: self.mouse.scroll(whole_units), "Kaydırma"
                ):
                    logger.debug("Kaydırma uygulandı (birim=%s).", whole_units)
                self._scroll_reference_y -= whole_units * sensitivity
        elif released:
            self._scroll_reference_y = None

    def _update_zoom(self, primary_landmarks, secondary_landmarks, frame_width, frame_height, now):
        """İkinci el yoksa veya sıkıştırılmamışsa yakınlaştırmayı devre dışı
        bırakır. İkinci el başparmak+işaret parmağını sıkıştırıp tuttuğunda,
        iki elin bilekleri arasındaki mesafenin değişimi yakınlaştırma
        miktarını belirler (eller birbirinden uzaklaşınca yakınlaştır,
        yaklaşınca uzaklaştır) — tıpkı bir dokunmatik yüzeydeki iki parmakla
        yakınlaştırma hareketinin, tüm ellerle büyütülmüş hâli gibi.
        """
        if secondary_landmarks is None:
            if self._zoom_gesture.active:
                self._zoom_gesture.force_release(now)
                self._zoom_reference_distance = None
            return

        def to_px(landmark):
            return (landmark.x * frame_width, landmark.y * frame_height)

        sec_thumb = to_px(secondary_landmarks[self.THUMB_TIP])
        sec_index = to_px(secondary_landmarks[self.INDEX_FINGER_TIP])
        pinch_distance = math.hypot(
            sec_thumb[0] - sec_index[0], sec_thumb[1] - sec_index[1]
        )

        triggered, released = self._zoom_gesture.update(pinch_distance, now)

        primary_wrist = to_px(primary_landmarks[self.WRIST])
        secondary_wrist = to_px(secondary_landmarks[self.WRIST])
        inter_hand_distance = math.hypot(
            primary_wrist[0] - secondary_wrist[0], primary_wrist[1] - secondary_wrist[1]
        )

        if triggered:
            self._zoom_reference_distance = inter_hand_distance
        elif self._zoom_gesture.active and self._zoom_reference_distance is not None:
            delta = inter_hand_distance - self._zoom_reference_distance
            sensitivity = max(1.0, self.config.zoom_sensitivity_px)
            whole_units = int(delta / sensitivity)
            if whole_units != 0:
                if self._safe_mouse_action(
                    lambda: self.mouse.zoom(whole_units), "Yakınlaştırma"
                ):
                    logger.debug("Yakınlaştırma uygulandı (birim=%s).", whole_units)
                self._zoom_reference_distance += whole_units * sensitivity
        elif released:
            self._zoom_reference_distance = None

    def apply_pinch_threshold(self, engage_px):
        """Tüm sıkıştırma hareketleri için tutturma eşiğini (ve buna bağlı
        histerezis bırakma eşiğini) çalışırken günceller. Panel kaydırıcısı
        bunu çağırır; config'i güncellemek TEK BAŞINA yeterli DEĞİLDİR
        çünkü her GestureState kendi eşik kopyasını tutar — bu yüzden dört
        hareketin de tanımını burada birlikte güncelliyoruz.
        """
        engage_px = max(1.0, float(engage_px))
        release_px = engage_px + self.PINCH_HYSTERESIS_GAP_PX
        self.config.pinch_engage_px = engage_px
        self.config.pinch_release_px = release_px
        for gesture in (
            self._left_gesture, self._right_gesture, self._scroll_gesture, self._zoom_gesture
        ):
            gesture.definition.engage_threshold_px = engage_px
            gesture.definition.release_threshold_px = release_px
        logger.info(
            "Sıkıştırma eşiği güncellendi (tutturma=%.1f px, bırakma=%.1f px).",
            engage_px, release_px,
        )

    def _release_all_gestures(self):
        """El kaybedildiğinde, duraklatıldığında veya durdurulduğunda,
        fare düğmesinin basılı takılı kalmaması için tüm etkin hareketleri
        güvenli şekilde sonlandırır.
        """
        now = time.monotonic()
        if self._left_gesture.force_release(now):
            self._safe_mouse_action(
                self.mouse.left_up, "Sol düğmeyi bırakma (el/izleme kaybedildi)"
            )
        self._right_gesture.force_release(now)
        self._scroll_gesture.force_release(now)
        self._scroll_reference_y = None
        self._zoom_gesture.force_release(now)
        self._zoom_reference_distance = None

    # --- Önizleme/ayarlar penceresi tarafından kullanılan genel arayüz ---
    # Aşağıdaki metodlar başka bir iş parçacığından (Tk ana iş parçacığı)
    # çağrılmak üzere tasarlanmıştır.

    def set_preview_active(self, active):
        """Önizleme penceresi görünür/gizli olduğunda çağrılır. Pencere
        kapalıyken önizleme karesi hazırlamanın getirdiği ek yükten kaçınmak
        için izleme döngüsü bu bayrağı kontrol eder.
        """
        if active:
            self._preview_active.set()
        else:
            self._preview_active.clear()
            try:
                while True:
                    self._preview_queue.get_nowait()
            except queue.Empty:
                pass

    def get_latest_preview(self):
        """En güncel önizleme karesini ve durum bilgisini döner.
        Hiçbir kare hazır değilse queue.Empty fırlatır.
        """
        return self._preview_queue.get_nowait()

    def apply_smoothing_params(self, mincutoff, beta):
        """Yumuşatma (One Euro Filter) parametrelerini çalışırken değiştirir
        (panodaki kaydırıcılardan çağrılır)."""
        mincutoff = max(0.05, float(mincutoff))
        beta = max(0.0, float(beta))
        self.config.smoothing_mincutoff = mincutoff
        self.config.smoothing_beta = beta
        self.smoother = CoordinateSmoother(mincutoff=mincutoff, beta=beta)
        logger.info(
            "Yumuşatma parametreleri güncellendi (mincutoff=%.2f, beta=%.2f).",
            mincutoff,
            beta,
        )

    def apply_detection_settings(self, detection_confidence, tracking_confidence):
        """El algılama/izleme güven eşiklerini çalışırken değiştirir. Bu,
        MediaPipe modelinin yeniden oluşturulmasını gerektirdiğinden diğer
        ayarlara göre daha maliyetlidir; bu yüzden panodaki kaydırıcılar
        bunu sürükleme sırasında değil, yalnızca fare bırakıldığında çağırır.
        """
        detection_confidence = min(max(float(detection_confidence), 0.1), 1.0)
        tracking_confidence = min(max(float(tracking_confidence), 0.1), 1.0)

        try:
            new_tracker = HandTracker(
                model_path=self.config.model_path,
                max_num_hands=1,
                detection_confidence=detection_confidence,
                presence_confidence=self.config.presence_confidence,
                tracking_confidence=tracking_confidence,
            )
        except Exception:
            logger.exception(
                "Yeni algılama ayarlarıyla model yeniden oluşturulamadı; "
                "önceki ayarlar korunuyor."
            )
            return

        with self._tracker_lock:
            old_tracker = self.hand_tracker
            self.hand_tracker = new_tracker
            old_tracker.close()

        self.config.detection_confidence = detection_confidence
        self.config.tracking_confidence = tracking_confidence
        self.smoother.reset()
        logger.info(
            "El algılama hassasiyeti güncellendi (algılama=%.2f, izleme=%.2f).",
            detection_confidence,
            tracking_confidence,
        )

    def set_low_light_enhancement(self, enabled):
        self.config.low_light_enhancement = bool(enabled)
        logger.info(
            "Düşük ışık iyileştirmesi %s.",
            "etkinleştirildi" if enabled else "devre dışı bırakıldı",
        )

    def set_two_hand_gestures_enabled(self, enabled):
        self.config.two_hand_gestures_enabled = bool(enabled)
        if not enabled:
            self._zoom_gesture.force_release(time.monotonic())
            self._zoom_reference_distance = None
        logger.info(
            "İki elle yakınlaştırma %s.",
            "etkinleştirildi" if enabled else "devre dışı bırakıldı",
        )

    def _build_preview_frame(self, frame_bgr, landmarks, secondary_landmarks=None):
        """Önizleme penceresinde göstermek üzere; aktif bölgeyi, ana elin
        referans noktalarını, dört hareket çizgisini (sol/sağ/kaydırma/
        yakınlaştırma — her biri etkin olduğunda farklı bir renge döner)
        ve varsa ikincil eli kareye çizer. Orijinal kareyi DEĞİŞTİRMEZ,
        bir kopyasını döner.
        """
        annotated = frame_bgr.copy()
        frame_height, frame_width = annotated.shape[:2]

        zone_x_min = int(frame_width * self.config.zone_margin_x)
        zone_x_max = int(frame_width * (1.0 - self.config.zone_margin_x))
        zone_y_min = int(frame_height * self.config.zone_margin_y)
        zone_y_max = int(frame_height * (1.0 - self.config.zone_margin_y))
        cv2.rectangle(
            annotated, (zone_x_min, zone_y_min), (zone_x_max, zone_y_max), (255, 200, 0), 2
        )

        if landmarks is not None:
            for landmark in landmarks:
                lx = int(landmark.x * frame_width)
                ly = int(landmark.y * frame_height)
                cv2.circle(annotated, (lx, ly), 3, (180, 180, 180), -1)

            def px(idx):
                lm = landmarks[idx]
                return int(lm.x * frame_width), int(lm.y * frame_height)

            thumb_pt = px(self.THUMB_TIP)
            index_pt = px(self.INDEX_FINGER_TIP)
            middle_pt = px(self.MIDDLE_FINGER_TIP)
            ring_pt = px(self.RING_FINGER_TIP)
            pinky_pt = px(self.PINKY_TIP)

            idle_color = (0, 200, 0)
            cv2.line(annotated, thumb_pt, middle_pt, (0, 0, 255) if self._left_gesture.active else idle_color, 2)
            cv2.line(annotated, thumb_pt, ring_pt, (0, 140, 255) if self._right_gesture.active else idle_color, 2)
            cv2.line(annotated, thumb_pt, pinky_pt, (255, 0, 255) if self._scroll_gesture.active else idle_color, 2)

            cv2.circle(annotated, index_pt, 8, (255, 120, 0), -1)   # imleç: turuncu-mavi
            cv2.circle(annotated, thumb_pt, 7, (0, 165, 255), -1)
            cv2.circle(annotated, middle_pt, 6, (0, 0, 255), -1)
            cv2.circle(annotated, ring_pt, 6, (0, 140, 255), -1)
            cv2.circle(annotated, pinky_pt, 6, (255, 0, 255), -1)

            if secondary_landmarks is not None:
                for landmark in secondary_landmarks:
                    sx = int(landmark.x * frame_width)
                    sy = int(landmark.y * frame_height)
                    cv2.circle(annotated, (sx, sy), 3, (140, 220, 140), -1)

                def spx(idx):
                    lm = secondary_landmarks[idx]
                    return int(lm.x * frame_width), int(lm.y * frame_height)

                sec_thumb_pt = spx(self.THUMB_TIP)
                sec_index_pt = spx(self.INDEX_FINGER_TIP)
                sec_wrist_pt = spx(self.WRIST)
                primary_wrist_pt = px(self.WRIST)

                zoom_color = (255, 255, 0) if self._zoom_gesture.active else idle_color
                cv2.line(annotated, sec_thumb_pt, sec_index_pt, zoom_color, 2)
                cv2.circle(annotated, sec_thumb_pt, 6, (0, 165, 255), -1)
                cv2.circle(annotated, sec_index_pt, 6, (255, 120, 0), -1)

                if self._zoom_gesture.active:
                    cv2.line(annotated, primary_wrist_pt, sec_wrist_pt, zoom_color, 2)

        return annotated

    def _push_preview_frame(self, frame_bgr, landmarks, secondary_landmarks=None):
        try:
            annotated = self._build_preview_frame(frame_bgr, landmarks, secondary_landmarks)
            now = time.monotonic()
            status = {
                "hand_detected": landmarks is not None,
                "second_hand_detected": secondary_landmarks is not None,
                "left_active": self._left_gesture.active,
                "is_dragging": self._left_gesture.held_seconds(now) >= self.DRAG_HOLD_SECONDS,
                "right_active": self._right_gesture.active,
                "scroll_active": self._scroll_gesture.active,
                "zoom_active": self._zoom_gesture.active,
                "is_paused": self.is_paused,
                "camera_connected": self.camera.is_connected,
                "performance": self.performance.snapshot(),
            }
            if self._preview_queue.full():
                try:
                    self._preview_queue.get_nowait()
                except queue.Empty:
                    pass
            self._preview_queue.put_nowait((annotated, status))
        except Exception:
            logger.debug("Önizleme karesi hazırlanırken hata oluştu.", exc_info=True)


class TrayApplication:
    """pystray tabanlı sistem tepsisi simgesi ve menüsünü yöneten sınıf."""

    ICON_SIZE = 64
    ACTIVE_COLOR = (46, 204, 113, 255)
    PAUSED_COLOR = (149, 165, 166, 255)
    GLYPH_COLOR = (33, 33, 33, 255)

    def __init__(self, controller, dashboard):
        self.controller = controller
        self.dashboard = dashboard
        self.icon = pystray.Icon(
            name="ElHareketiFareKontrolu",
            icon=self._build_icon_image(paused=False),
            title="El Hareketiyle Fare Kontrolü",
            menu=self._build_menu(),
        )

    def _build_menu(self):
        return pystray.Menu(
            pystray.MenuItem("El Hareketiyle Fare Kontrolü", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Paneli Göster", self._on_show_dashboard),
            pystray.MenuItem(
                lambda item: "Devam Ettir" if self.controller.is_paused else "Duraklat",
                self._on_toggle_pause,
            ),
            pystray.MenuItem("Çıkış", self._on_quit),
        )

    def _build_icon_image(self, paused):
        size = self.ICON_SIZE
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        color = self.PAUSED_COLOR if paused else self.ACTIVE_COLOR
        margin = 6
        draw.ellipse((margin, margin, size - margin, size - margin), fill=color)

        if paused:
            draw.rectangle((24, 18, 30, 46), fill=self.GLYPH_COLOR)
            draw.rectangle((34, 18, 40, 46), fill=self.GLYPH_COLOR)
        else:
            draw.ellipse((26, 26, 38, 38), fill=self.GLYPH_COLOR)

        return image

    def _on_toggle_pause(self, icon, item):
        self.controller.toggle_pause()
        self.refresh()

    def _on_show_dashboard(self, icon, item):
        self.dashboard.request_show()

    def _on_quit(self, icon, item):
        logger.info("Kullanıcı sistem tepsisinden çıkışı seçti.")
        # Tkinter yalnızca kendi ana iş parçacığından güvenle güncellenebildiği
        # için kapatma isteğini doğrudan burada değil, panelin iş parçacığı
        # açısından güvenli komut kuyruğu üzerinden iletiyoruz.
        self.dashboard.request_quit()

    def refresh(self):
        try:
            self.icon.icon = self._build_icon_image(self.controller.is_paused)
            self.icon.update_menu()
        except Exception:
            logger.debug(
                "Sistem tepsisi simgesi güncellenirken hata oluştu.", exc_info=True
            )


class DashboardWindow:
    """Uygulamanın ana görsel arayüzü: kamera önizlemesi (el iskeleti,
    aktif bölge ve üç hareket çizgisi çizilmiş hâlde) ile canlı ayar
    kaydırıcılarını bir arada gösteren panel.

    Panel VARSAYILAN OLARAK GÖRÜNÜRDÜR (kullanıcı bunu tepsi yerine sürekli
    görünen bir pano olarak istedi). Kapatma (X) düğmesi uygulamayı
    SONLANDIRMAZ, yalnızca paneli sistem tepsisine küçültür; uygulamadan
    tamamen çıkmak için tepsi menüsündeki "Çıkış" kullanılmalıdır.

    Diğer iş parçacıklarından (pystray, izleme iş parçacığı) bu sınıfa
    yalnızca `request_show()` ve `request_quit()` üzerinden, iş parçacığı
    açısından güvenli bir kuyruk aracılığıyla erişilmelidir; Tkinter
    nesneleri yalnızca bu paneli oluşturan ana iş parçacığından (Tk
    `mainloop`'un çalıştığı iş parçacığından) değiştirilmelidir.
    """

    PREVIEW_WIDTH = 480
    PREVIEW_HEIGHT = 360
    TICK_INTERVAL_MS = 33  # ~30 kare/saniye

    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.quit_callback = None  # main() tarafından atanır

        self._visible = False
        self._photo_image = None  # Tkinter'ın görüntüyü elden çıkarmaması için referans
        self._command_queue = queue.Queue()

        self.root.title("El Hareketiyle Fare Kontrolü — Kontrol Paneli")
        self.root.resizable(False, False)
        # X düğmesi: uygulamayı kapatmaz, yalnızca tepsiye küçültür.
        self.root.protocol("WM_DELETE_WINDOW", self.hide)

        self._build_widgets()
        self.show()
        self._schedule_tick()

    # --- Arayüz kurulumu ---

    def _build_widgets(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            container,
            width=self.PREVIEW_WIDTH,
            height=self.PREVIEW_HEIGHT,
            background="#202020",
            highlightthickness=0,
        )
        self.canvas.pack(pady=(0, 8))
        self.canvas.create_text(
            self.PREVIEW_WIDTH // 2,
            self.PREVIEW_HEIGHT // 2,
            text="Önizleme bekleniyor...",
            fill="#aaaaaa",
            font=("Arial", 12),
            tags=("placeholder",),
        )

        self.hero_frame = tk.Frame(container, background="#e5e5e5")
        self.hero_frame.pack(fill="x", pady=(0, 4))
        self.hero_label = tk.Label(
            self.hero_frame, text="⚪ BEKLENİYOR", font=("Arial", 16, "bold"),
            background="#e5e5e5", foreground="#333333", pady=8,
        )
        self.hero_label.pack(fill="x")
        self.hero_sub_label = ttk.Label(container, text="", foreground="#777777")
        self.hero_sub_label.pack(anchor="w", pady=(0, 8))

        info_row = ttk.Frame(container)
        info_row.pack(fill="x", pady=(0, 8))

        status_frame = ttk.LabelFrame(info_row, text="SYSTEM STATUS", padding=6)
        status_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        self.status_active_label = ttk.Label(status_frame, text="Durum: —")
        self.status_active_label.pack(anchor="w")
        self.status_camera_label = ttk.Label(status_frame, text="Kamera: —")
        self.status_camera_label.pack(anchor="w")
        self.status_hand_label = ttk.Label(status_frame, text="El: —")
        self.status_hand_label.pack(anchor="w")
        self.status_quality_label = ttk.Label(status_frame, text="Takip kalitesi: —")
        self.status_quality_label.pack(anchor="w")

        perf_frame = ttk.LabelFrame(info_row, text="PERFORMANCE", padding=6)
        perf_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.perf_fps_label = ttk.Label(perf_frame, text="FPS: —")
        self.perf_fps_label.pack(anchor="w")
        self.perf_latency_label = ttk.Label(perf_frame, text="Gecikme: —")
        self.perf_latency_label.pack(anchor="w")
        self.perf_confidence_label = ttk.Label(perf_frame, text="El güveni: —")
        self.perf_confidence_label.pack(anchor="w")
        self.perf_dropped_label = ttk.Label(perf_frame, text="Düşen kare: —")
        self.perf_dropped_label.pack(anchor="w")

        legend = ttk.Label(
            container,
            text=(
                "Turuncu: imleç (işaret parmağı) · Kırmızı: sol tık/sürükle "
                "(başparmak+orta) · Turuncu-sarı: sağ tık (başparmak+yüzük) · "
                "Mor: kaydırma (başparmak+serçe) · Sarı: ikinci elle "
                "yakınlaştırma (ikinci elde başparmak+işaret parmağı, "
                "eller birbirinden uzaklaştırılıp yaklaştırılır)"
            ),
            foreground="#777777",
            wraplength=self.PREVIEW_WIDTH,
            justify="left",
        )
        legend.pack(anchor="w", pady=(0, 8))

        config = self.controller.config
        self.pinch_var = tk.DoubleVar(value=config.pinch_engage_px)
        self.zone_var = tk.DoubleVar(value=config.zone_margin_x)
        self.responsiveness_var = tk.DoubleVar(value=config.smoothing_beta)
        self.detection_var = tk.DoubleVar(value=config.detection_confidence)
        self.tracking_var = tk.DoubleVar(value=config.tracking_confidence)
        self.low_light_var = tk.BooleanVar(value=config.low_light_enhancement)
        self.two_hand_var = tk.BooleanVar(value=config.two_hand_gestures_enabled)

        self._add_slider(
            container, "Sıkıştırma Hassasiyeti", self.pinch_var, 15, 70,
            command=self._on_pinch_threshold_change, value_format="{:.0f} px",
        )
        self._add_slider(
            container, "Aktif Bölge Marjı", self.zone_var, 0.05, 0.45,
            command=self._on_zone_change, value_format="%{:.0f}", value_scale=100,
        )
        self._add_slider(
            container, "Tepki Hızı", self.responsiveness_var, 0.0, 1.5,
            command=self._on_responsiveness_change, value_format="{:.2f}",
        )
        self._add_slider(
            container, "El Algılama Güveni", self.detection_var, 0.1, 1.0,
            command=None, value_format="{:.2f}",
            release_callback=self._on_detection_settings_change,
        )
        self._add_slider(
            container, "İzleme Güveni", self.tracking_var, 0.1, 1.0,
            command=None, value_format="{:.2f}",
            release_callback=self._on_detection_settings_change,
        )

        ttk.Checkbutton(
            container,
            text="Düşük ışıkta otomatik görüntü iyileştirme",
            variable=self.low_light_var,
            command=self._on_low_light_toggle,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Checkbutton(
            container,
            text="İkinci elle yakınlaştırma hareketini etkinleştir",
            variable=self.two_hand_var,
            command=self._on_two_hand_toggle,
        ).pack(anchor="w", pady=(0, 8))

        note = ttk.Label(
            container,
            text=(
                "Not: Aktif bölge marjı yatay/dikey eksende birlikte değişir. "
                "\"Tepki Hızı\" yükseldikçe hızlı hareketlerde gecikme azalır."
            ),
            foreground="#777777",
            wraplength=self.PREVIEW_WIDTH,
            justify="left",
        )
        note.pack(anchor="w", pady=(0, 8))

        button_row = ttk.Frame(container)
        button_row.pack(fill="x")

        self.pause_button = ttk.Button(
            button_row, text="Duraklat", command=self._on_pause_toggle
        )
        self.pause_button.pack(side="left")

        ttk.Button(
            button_row, text="Tepsiye Küçült", command=self.hide
        ).pack(side="right")

    def _add_slider(
        self, parent, label_text, var, from_, to,
        command, value_format="{:.2f}", value_scale=1, release_callback=None,
    ):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)

        ttk.Label(row, text=label_text, width=18, anchor="w").pack(side="left")
        value_label = ttk.Label(row, text="", width=9, anchor="e")
        value_label.pack(side="right")

        def refresh_value_label():
            value_label.config(text=value_format.format(var.get() * value_scale))

        def on_move(_value):
            refresh_value_label()
            if command is not None:
                command(var.get())

        scale = ttk.Scale(
            row, from_=from_, to=to, orient="horizontal", variable=var, command=on_move
        )
        scale.pack(side="left", fill="x", expand=True, padx=6)

        if release_callback is not None:
            scale.bind("<ButtonRelease-1>", lambda _evt: release_callback())

        refresh_value_label()

    # --- Kaydırıcı/onay kutusu geri çağrıları (Tk ana iş parçacığında çalışır) ---

    def _on_pinch_threshold_change(self, value):
        self.controller.apply_pinch_threshold(float(value))

    def _on_zone_change(self, value):
        value = float(value)
        self.controller.config.zone_margin_x = value
        self.controller.config.zone_margin_y = value

    def _on_responsiveness_change(self, value):
        self.controller.apply_smoothing_params(
            mincutoff=self.controller.config.smoothing_mincutoff, beta=float(value)
        )

    def _on_detection_settings_change(self):
        self.controller.apply_detection_settings(
            detection_confidence=self.detection_var.get(),
            tracking_confidence=self.tracking_var.get(),
        )

    def _on_low_light_toggle(self):
        self.controller.set_low_light_enhancement(self.low_light_var.get())

    def _on_two_hand_toggle(self):
        self.controller.set_two_hand_gestures_enabled(self.two_hand_var.get())

    def _on_pause_toggle(self):
        self.controller.toggle_pause()
        self._refresh_pause_button_text()

    # --- Diğer iş parçacıklarından çağrılabilecek, güvenli genel arayüz ---

    def request_show(self):
        """pystray'in kendi iş parçacığından çağrılması güvenlidir."""
        self._command_queue.put("show")

    def request_quit(self):
        """pystray'in kendi iş parçacığından çağrılması güvenlidir."""
        self._command_queue.put("quit")

    # --- Görünürlük (yalnızca Tk ana iş parçacığından çağrılmalıdır) ---

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self._visible = True
        self.controller.set_preview_active(True)
        self._refresh_pause_button_text()

    def hide(self):
        self.root.withdraw()
        self._visible = False
        self.controller.set_preview_active(False)

    def _refresh_pause_button_text(self):
        self.pause_button.config(
            text="Devam Ettir" if self.controller.is_paused else "Duraklat"
        )

    # --- Periyodik döngü: hem komut kuyruğunu hem de önizleme karesini işler ---

    def _schedule_tick(self):
        self.root.after(self.TICK_INTERVAL_MS, self._tick)

    def _tick(self):
        while True:
            try:
                command = self._command_queue.get_nowait()
            except queue.Empty:
                break

            if command == "show":
                self.show()
            elif command == "quit":
                self._perform_quit()
                return  # Kapanış başladı; yeniden zamanlama yapma.

        if self._visible:
            self._update_preview_frame()
            self._refresh_pause_button_text()

        self._schedule_tick()

    def _update_preview_frame(self):
        try:
            annotated_bgr, status = self.controller.get_latest_preview()
        except queue.Empty:
            return

        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb).resize(
            (self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
        )
        self._photo_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_image)

        self._update_hero(status)
        self._update_system_status(status)
        self._update_performance(status.get("performance", {}))

    def _update_hero(self, status):
        """Büyük "ŞU ANDA NE YAPIYORSUN" göstergesi. Öncelik sırası, hareket
        motorunun öncelik sırasıyla aynıdır (duraklatma/kamera/el yokluğu
        her zaman hareketlerin önünde gelir)."""
        if status["is_paused"]:
            text, sub, color = "⏸ DURAKLATILDI", "Panelden veya kısayoldan devam edin", "#f0e0a0"
        elif not status["camera_connected"]:
            text, sub, color = "🔌 KAMERA BAĞLI DEĞİL", "Yeniden bağlanmaya çalışılıyor...", "#f0a0a0"
        elif not status["hand_detected"]:
            text, sub, color = "⚪ EL ALGILANMADI", "Elinizi kameraya gösterin", "#e5e5e5"
        elif status["left_active"]:
            if status.get("is_dragging"):
                text, sub, color = "🟢 SÜRÜKLENİYOR", "Başparmak + Orta Parmak (basılı)", "#a0e0a0"
            else:
                text, sub, color = "🔴 SOL TIK", "Başparmak + Orta Parmak", "#f0a0a0"
        elif status["right_active"]:
            text, sub, color = "🟡 SAĞ TIK", "Başparmak + Yüzük Parmağı", "#f0d888"
        elif status["scroll_active"]:
            text, sub, color = "🟣 KAYDIRILIYOR", "Başparmak + Serçe Parmak", "#d0a0e0"
        elif status["zoom_active"]:
            text, sub, color = "🔵 YAKINLAŞTIRILIYOR", "İkinci el: Başparmak + İşaret Parmağı", "#a0c0f0"
        else:
            text, sub, color = "🟠 İMLEÇ", "İşaret parmağıyla imleci kontrol edin", "#f5c99a"

        if status.get("second_hand_detected") and not status["is_paused"] and status["hand_detected"]:
            sub += " · 2 el algılandı"

        self.hero_label.config(text=text, background=color)
        self.hero_frame.config(background=color)
        self.hero_sub_label.config(text=sub)

    def _update_system_status(self, status):
        self.status_active_label.config(
            text="Durum: " + ("Duraklatıldı" if status["is_paused"] else "Aktif")
        )
        self.status_camera_label.config(
            text="Kamera: " + ("Bağlı" if status["camera_connected"] else "BAĞLI DEĞİL")
        )
        if status["hand_detected"]:
            hand_text = "2 el" if status.get("second_hand_detected") else "1 el"
        else:
            hand_text = "Yok"
        self.status_hand_label.config(text="El: " + hand_text)

        confidence = status.get("performance", {}).get("hand_confidence", 0.0)
        if not status["hand_detected"]:
            quality_text = "—"
        elif confidence >= 0.85:
            quality_text = "Mükemmel"
        elif confidence >= 0.6:
            quality_text = "İyi"
        elif confidence >= 0.3:
            quality_text = "Orta"
        else:
            quality_text = "Zayıf"
        self.status_quality_label.config(text="Takip kalitesi: " + quality_text)

    def _update_performance(self, perf):
        """Panelin PERFORMANCE bloğu: hepsi GERÇEK ölçümlerden gelir (bkz.
        PerformanceMonitor) — hiçbir sayı sabit/uydurma değildir. Örnek
        sayısı henüz azsa (uygulama yeni başladıysa) değerler 0 görünebilir;
        bu bir hata değil, henüz yeterli ölçüm birikmediğinin göstergesidir.
        """
        self.perf_fps_label.config(text="FPS: {:.0f}".format(perf.get("fps", 0.0)))
        self.perf_latency_label.config(
            text="Gecikme: kare {:.0f} ms · izleme {:.0f} ms".format(
                perf.get("frame_time_ms", 0.0), perf.get("tracking_latency_ms", 0.0)
            )
        )
        self.perf_confidence_label.config(
            text="El güveni: {:.0f}%".format(perf.get("hand_confidence", 0.0) * 100)
        )
        self.perf_dropped_label.config(
            text="Düşen kare: {}".format(perf.get("dropped_frames", 0))
        )

    def _perform_quit(self):
        logger.info("Kapatma isteği kontrol paneli döngüsünde işleniyor.")
        if self.quit_callback is not None:
            self.quit_callback()


def main():
    config = AppConfig()
    configure_logging(config.log_file_path)

    logger.info("=" * 60)
    logger.info("El Hareketiyle Fare Kontrolü uygulaması başlatılıyor.")
    logger.info("Günlük dosyası: %s", config.log_file_path)
    logger.info("=" * 60)

    controller = None
    worker_thread = None
    root = None

    try:
        controller = HandGestureMouseController(config)

        # Tkinter kök penceresi: ana iş parçacığında oluşturulur. Panel,
        # kullanıcının tercihi doğrultusunda varsayılan olarak GÖRÜNÜRDÜR
        # (DashboardWindow.__init__ içinde show() çağrılır); tepsiye
        # küçültmek için pencerenin X düğmesi veya "Tepsiye Küçült" kullanılır.
        root = tk.Tk()
        dashboard = DashboardWindow(root, controller)

        tray = TrayApplication(controller, dashboard)
        controller.on_state_change = lambda paused: tray.refresh()

        hotkeys = GlobalHotkeyManager({
            config.hotkey_toggle_pause: controller.toggle_pause,
            config.hotkey_show_dashboard: dashboard.request_show,
        })

        worker_thread = threading.Thread(
            target=controller.run, name="ElTakipIsParcacigi", daemon=True
        )

        def shutdown():
            try:
                hotkeys.stop()
            except Exception:
                logger.debug("Klavye kısayolları durdurulurken hata oluştu.", exc_info=True)
            try:
                controller.stop()
                worker_thread.join(timeout=5.0)
            except Exception:
                logger.exception("Kontrolcü durdurulurken hata oluştu.")
            try:
                tray.icon.stop()
            except Exception:
                logger.debug("Tepsi simgesi durdurulurken hata oluştu.", exc_info=True)
            try:
                root.quit()
            except Exception:
                logger.debug("Tk ana döngüsü durdurulurken hata oluştu.", exc_info=True)

        dashboard.quit_callback = shutdown

        worker_thread.start()
        hotkeys.start()  # pynput yoksa/başarısız olursa sessizce devre dışı kalır, uygulama etkilenmez

        # pystray'in simge döngüsü kendi arka plan iş parçacığında çalışır;
        # bu, Tkinter'in ana döngüsünün ana iş parçacığını kullanmasına izin
        # verir (özellikle macOS'ta ikisinin de ana iş parçacığını istemesi
        # çakışmasını önlemek için pystray'in resmi olarak önerdiği yöntem).
        tray.icon.run_detached()

        root.mainloop()

    except KeyboardInterrupt:
        logger.info("Klavyeden kesme sinyali alındı (Ctrl+C).")
    except Exception:
        logger.exception("Uygulama beklenmeyen bir hatayla karşılaştı ve kapanıyor.")
    finally:
        if controller is not None:
            controller.stop()
        if worker_thread is not None:
            worker_thread.join(timeout=5.0)
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass
        logger.info("Uygulama kapatıldı.")


if __name__ == "__main__":
    main()
