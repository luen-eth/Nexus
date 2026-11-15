# Bağımsız K-quant oracle fixture'ları

İki blok ve değişken scale/min/nibble içerir. Expected F32 dosyaları değiştirilmemiş resmi `gguf.quants.dequantize` ile üretildi: ggml-org/llama.cpp `fc2b0053ffe878ff5a26934bdb555681f15bc699` (29 Nisan 2026 öncesi). Nexus yazıcısı ya da decoder'ı üretimde kullanılmadı. Upstream kaynak ve MIT lisansı bu sabit commit altında incelenebilir: https://github.com/ggml-org/llama.cpp/tree/fc2b0053ffe878ff5a26934bdb555681f15bc699/gguf-py. Binary bloklar sentetik test verileridir; model ağırlığı içermez.

Testteki encoder oracle formülleri de aynı sürümün C `dequantize_row_q4_K` / `dequantize_row_q2_K` düzenini izler; encoder'dan çıkan byte'ların input'a yakınlığını Nexus decoder'ını çağırmadan ölçer.
