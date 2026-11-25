"""16-bit original data preservation and display/model preprocessing separation."""
import numpy as np
import tifffile

from gistudio.core.dataset_import import ImportOptions, import_folder
from gistudio.core.image_io import file_sha256, load_mono, read_image, write_image_lossless, estimate_effective_bits
from gistudio.core.preprocessing import DisplayTransform, InputNormalization


def test_16bit_tiff_and_png_roundtrip_exact(tmp_path):
    rng = np.random.default_rng(1)
    a12 = rng.integers(0, 4096, (37, 53), dtype=np.uint16)        # 12-bit data in 16-bit container
    a16 = rng.integers(0, 65536, (37, 53), dtype=np.uint16)       # full 16-bit
    tifffile.imwrite(tmp_path / "a12.tif", a12)
    write_image_lossless(tmp_path / "a16.png", a16)
    r12, r16 = read_image(tmp_path / "a12.tif"), read_image(tmp_path / "a16.png")
    assert r12.dtype == np.uint16 and np.array_equal(r12, a12)
    assert r16.dtype == np.uint16 and np.array_equal(r16, a16)
    assert estimate_effective_bits(a12) <= 12


def test_unicode_path_png(tmp_path):
    a = np.arange(100, dtype=np.uint8).reshape(10, 10)
    p = tmp_path / "유리병_테스트.png"
    write_image_lossless(p, a)
    assert np.array_equal(read_image(p), a)


def test_import_never_modifies_originals(project, tmp_path):
    d = tmp_path / "raw" / "B1" / "S1"; d.mkdir(parents=True)
    a = np.random.default_rng(0).integers(0, 65535, (20, 30), dtype=np.uint16)
    tifffile.imwrite(d / "f_a0.tif", a)
    before = file_sha256(d / "f_a0.tif"); mtime = (d / "f_a0.tif").stat().st_mtime_ns
    rep = import_folder(project, tmp_path / "raw", ImportOptions())
    assert rep.imported == 1
    f = project.all_frames()[0]
    assert f["sha256"] == before and f["dtype"] == "uint16"
    img = load_mono(f["path"])
    assert not img.flags.writeable              # cached originals are immutable
    disp = DisplayTransform(stretch_percentiles=(1, 99), gamma=0.7, clahe=True, invert=True).apply(img)
    assert disp.dtype == np.uint8 and np.array_equal(img, a)   # display transform did not touch the original
    x = InputNormalization(bit_depth=16).apply(img)
    assert x.dtype == np.float32 and np.allclose(x * 65535, a, atol=0.01)
    assert file_sha256(d / "f_a0.tif") == before and (d / "f_a0.tif").stat().st_mtime_ns == mtime


def test_window_level_and_bitdepth_normalization():
    a = np.array([[0, 1023, 4095]], np.uint16)
    d = DisplayTransform(bit_depth=12).apply(a)
    assert d.tolist() == [[0, 64, 255]]
    d2 = DisplayTransform(window=100, level=1023).apply(a)
    assert d2[0, 0] == 0 and d2[0, 2] == 255 and 120 <= d2[0, 1] <= 135
    n = InputNormalization(bit_depth=12).apply(a)
    assert np.isclose(n[0, 2], 1.0) and np.isclose(n[0, 1], 1023 / 4095)


def test_duplicate_detection(project, tmp_path):
    d = tmp_path / "raw" / "B1" / "S1"; d.mkdir(parents=True)
    a = np.ones((8, 8), np.uint16)
    tifffile.imwrite(d / "x_a0.tif", a); tifffile.imwrite(d / "y_a10.tif", a)
    rep = import_folder(project, tmp_path / "raw", ImportOptions())
    assert rep.imported == 1 and len(rep.duplicates) == 1


def test_angle_mapping_regex_csv_and_natural_sort(project, tmp_path):
    d = tmp_path / "raw" / "B1" / "S1"; d.mkdir(parents=True)
    for i in (1, 2, 10):
        tifffile.imwrite(d / f"frame_{i}.tif", np.full((4, 4), i, np.uint16))
    (d / "angles.csv").write_text("filename,angle\nframe_1.tif,0\nframe_2.tif,120\nframe_10.tif,240\n", encoding="utf-8")
    import_folder(project, tmp_path / "raw", ImportOptions())
    fr = project.frames(project.sessions(project.bottles()[0]["id"])[0]["id"])
    assert [f["angle_deg"] for f in fr] == [0.0, 120.0, 240.0]
    assert [f["frame_index"] for f in fr] == [1, 2, 10]
