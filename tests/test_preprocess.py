import pandas as pd
import pytest
from PIL import Image

from src.common import schema
from src.preprocess import metadata, postprocess, split
from src.preprocess.make_dummy import make
from src.preprocess.transforms import estimate_jpeg_quality

CFG = {"jpeg_quality": 95, "kakao": {"long_side": 320, "quality": 70}, "resize": {"short_side": 128}}


@pytest.fixture(scope="module")
def dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("dummy")
    make(root, per_object=4)
    meta = root / "metadata"
    rows = split.build_rows(
        pd.read_csv(meta / "real_metadata.csv", dtype=str, keep_default_na=False),
        pd.read_csv(meta / "fake_mapping.csv", dtype=str, keep_default_na=False),
    )
    return root, rows


# ---- postprocess ----

def test_postprocess_outputs_same_format_for_real_and_fake(dummy):
    root, _ = dummy
    data = root / "data"
    written = postprocess.run(data, CFG, postprocess.AUTO_VARIANTS)
    n_sources = len(postprocess.find_sources(data))
    assert all(n == n_sources for n in written.values())

    real = data / "processed" / "original" / "jaehwan_mouse_001.jpg"
    fake = data / "processed" / "original" / "fakeA_jaehwan_mouse_001_s1.jpg"
    for p in (real, fake):
        with Image.open(p) as im:
            assert im.format == "JPEG"
            assert not im.getexif()  # 진짜의 EXIF가 남으면 모델이 그걸로 맞힌다


def test_postprocess_sizes_and_quality(dummy):
    root, _ = dummy
    out = root / "data" / "processed"
    with Image.open(out / "kakao" / "jaehwan_mouse_001.jpg") as im:
        assert max(im.size) == 320
    with Image.open(out / "resize" / "jaehwan_mouse_001.jpg") as im:
        assert min(im.size) == 128
    assert abs(estimate_jpeg_quality(out / "kakao" / "jaehwan_mouse_001.jpg") - 70) <= 2
    assert abs(estimate_jpeg_quality(out / "original" / "jaehwan_mouse_001.jpg") - 95) <= 2


def test_postprocess_skips_existing(dummy):
    root, _ = dummy
    written = postprocess.run(root / "data", CFG, postprocess.AUTO_VARIANTS)
    assert sum(written.values()) == 0


def test_check_real_files_flags_non_jpg(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    Image.new("RGB", (32, 32)).save(real / "taei_mouse_001.jpg", "JPEG")
    (real / "taei_mouse_002.heic").write_bytes(b"\x00\x00\x00\x18ftypheic")  # 아이폰 기본 형식
    Image.new("RGB", (32, 32)).save(real / "taei_mouse_003.jpg", "PNG")  # 확장자만 .jpg
    (real / ".DS_Store").write_bytes(b"")
    problems = postprocess.check_real_files(tmp_path)
    assert len(problems) == 2
    assert problems[0].startswith("taei_mouse_002.heic") and problems[1].startswith("taei_mouse_003.jpg")
    assert postprocess.check_real_files(tmp_path.parent / "nothing") == []


def test_iphone_jpg_is_accepted(tmp_path):
    # 아이폰 JPG에는 HDR 보조 이미지가 붙어 있어 Pillow가 MPO로 읽는다
    real = tmp_path / "real"
    real.mkdir()
    Image.new("RGB", (40, 30)).save(
        real / "taei_mouse_001.jpg", "MPO", save_all=True, append_images=[Image.new("RGB", (20, 15))]
    )
    with Image.open(real / "taei_mouse_001.jpg") as im:
        assert im.format == "MPO"
    assert postprocess.check_real_files(tmp_path) == []
    assert postprocess.run(tmp_path, CFG, ["original"]) == {"original": 1}
    with Image.open(tmp_path / "processed" / "original" / "taei_mouse_001.jpg") as im:
        assert (im.format, im.size) == ("JPEG", (40, 30))


def test_real_photos_in_drive_folders(tmp_path):
    # 드라이브: 사람별 하위 폴더, 구글 폼이 붙인 " - 이름", 대문자 이름·확장자
    real = tmp_path / "real"
    for d in ("taei(tripod,desk)", "Yoojaehwan(mouse)", ".trash"):
        (real / d).mkdir(parents=True)
    exif = Image.Exif()
    exif[0x0110] = "iPhone Air"
    Image.new("RGB", (40, 30)).save(real / "taei(tripod,desk)" / "taei_tripod_001 - 김태이.JPG", exif=exif)
    Image.new("RGB", (40, 30)).save(real / "Yoojaehwan(mouse)" / "Yoojaehwan_mouse_nomemo_001.jpg", exif=exif)
    Image.new("RGB", (40, 30)).save(real / ".trash" / "taei_tripod_002.jpg")  # 숨김 폴더는 무시
    meta, errors, warnings = metadata.build(tmp_path, objects={"tripod", "mouse"})
    assert errors == [] and warnings == []
    assert sorted(meta["filename"]) == ["taei_tripod_001.jpg", "yoojaehwan_mouse_nomemo_001.jpg"]
    assert sorted(meta["photographer"]) == ["taei", "yoojaehwan"]
    sources = postprocess.find_sources(tmp_path)
    assert set(sources) == {"taei_tripod_001", "yoojaehwan_mouse_nomemo_001"}
    assert sources["taei_tripod_001"].name == "taei_tripod_001 - 김태이.JPG"


def test_same_photo_uploaded_twice_is_reported(tmp_path):
    # 폼 응답 폴더의 사진을 real에 또 올린 경우: 규칙상 이름이 같아진다
    real = tmp_path / "real"
    for d in ("a", "b"):
        (real / d).mkdir(parents=True)
    Image.new("RGB", (40, 30)).save(real / "a" / "taei_mouse_001.jpg")
    Image.new("RGB", (40, 30)).save(real / "b" / "taei_mouse_001 - 김태이.jpg")
    meta, errors, _ = metadata.build(tmp_path)
    assert len(meta) == 1 and len(errors) == 1 and "겹침" in errors[0]
    with pytest.raises(ValueError, match="stem 중복"):
        postprocess.find_sources(tmp_path)


def test_recapture_matches_by_stem(dummy, tmp_path):
    root, _ = dummy
    Image.new("RGB", (64, 64)).save(tmp_path / "jaehwan_mouse_001.jpg")
    Image.new("RGB", (64, 64)).save(tmp_path / "IMG_1234.jpg")
    n, unknown = postprocess.ingest_recapture(root / "data", tmp_path, CFG)
    assert n == 1 and unknown == ["IMG_1234.jpg"]


# ---- split ----

def test_build_has_no_leakage(dummy):
    _, rows = dummy
    result, skipped = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    assert split.check(result) == []
    assert set(result["split"]) == {"train", "val", "test"}
    assert len(result) + len(skipped) == len(rows)


def test_reinforcement_only_goes_to_train(dummy):
    _, rows = dummy
    result, skipped = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    reinforced = rows.loc[rows["is_reinforcement"] == "1", "filename"]
    in_result = result[result["filename"].isin(reinforced)]
    assert (in_result["split"] == "train").all()
    assert len(skipped) > 0  # val/test 원본에서 나온 보강분은 빠져야 함
    assert set(skipped["filename"]) == set(reinforced) - set(in_result["filename"])


def test_test_photographers_are_held_out(dummy):
    _, rows = dummy
    result, _ = split.build(rows, [0.7, 0.15, 0.15], seed=42, test_photographers=["taei"])
    taei_objects = set(rows.loc[rows["photographer"] == "taei", "object"])
    assert (result.loc[result["object"].isin(taei_objects), "split"] == "test").all()
    assert split.check(result) == []


def test_append_keeps_existing_rows_fixed(dummy):
    _, rows = dummy
    base, _ = split.build(rows.iloc[: len(rows) // 2], [0.7, 0.15, 0.15], seed=0)
    result, skipped = split.append(base, rows)
    merged = result.merge(base, on="filename", suffixes=("", "_before"))
    assert (merged["split"] == merged["split_before"]).all()
    added = result[~result["filename"].isin(base["filename"])]
    assert (added["split"] == "train").all()
    assert len(added) + len(skipped) == len(rows) - len(base)
    assert split.check(result) == []


def test_check_detects_group_leakage(dummy):
    _, rows = dummy
    result, _ = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    fake_row = result[result["fake_type"] == "A"].index[0]
    result.loc[fake_row, "split"] = "test" if result.loc[fake_row, "split"] != "test" else "train"
    errors = split.check(result)
    assert any("원본 그룹" in e for e in errors)


def test_fake_with_unknown_source_is_rejected(dummy):
    root, _ = dummy
    meta = root / "metadata"
    real = pd.read_csv(meta / "real_metadata.csv", dtype=str, keep_default_na=False)
    fake = pd.read_csv(meta / "fake_mapping.csv", dtype=str, keep_default_na=False)
    fake.loc[0, "source_filename"] = "nobody_nothing_999.jpg"
    with pytest.raises(ValueError, match="real_metadata"):
        split.build_rows(real, fake)


def test_fake_a_from_nomemo_source_is_rejected(dummy):
    root, _ = dummy
    meta = root / "metadata"
    real = pd.read_csv(meta / "real_metadata.csv", dtype=str, keep_default_na=False)
    fake = pd.read_csv(meta / "fake_mapping.csv", dtype=str, keep_default_na=False)
    row = fake.index[fake["source_filename"] == "taei_keyboard_nomemo_001.jpg"][0]  # 더미에서는 B
    fake.loc[row, ["fake_type", "fake_filename"]] = ["A", "fakeA_taei_keyboard_nomemo_001_s1.png"]
    with pytest.raises(ValueError, match="메모 없는 원본"):
        split.build_rows(real, fake)


def test_nomemo_rows_follow_their_source(dummy):
    _, rows = dummy
    result, _ = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    nomemo = result[~result["filename"].map(schema.has_memo)]
    assert set(nomemo["fake_type"]) == {"real", "B", "C"}  # 메모 없는 가짜는 B·C만
    fake_b = nomemo[nomemo["fake_type"] == "B"].set_index("group_id")["split"]
    real = nomemo[nomemo["fake_type"] == "real"].set_index("group_id")["split"]
    assert fake_b.index.isin(real.index).all() and (fake_b == real[fake_b.index]).all()


# ---- export (모델 1 형식) ----

def _model1_errors(df, leakage_columns=("item_id",)):
    """모델 1(Model_A src/dataset.py read_metadata)의 이미지 외 검사를 그대로 옮긴 것."""
    errors = []
    for col in ["image_path", "label", "fake_type", "source_id", "split"]:
        if col not in df or df[col].astype(str).str.strip().eq("").any():
            errors.append(f"필수 컬럼 없음/빈값: {col}")
    labels = pd.to_numeric(df["label"], errors="coerce")
    ok = ((labels == 0) & (df["fake_type"] == "real")) | ((labels == 1) & df["fake_type"].isin(["A", "B", "C", "unknown"]))
    if not ok.all():
        errors.append("label/fake_type 오류")
    for col in ["source_id", *leakage_columns]:
        present = df[df[col].astype(str).str.strip() != ""]
        if (present.groupby(col)["split"].nunique() > 1).any():
            errors.append(f"누수: {col}")
    if df["image_path"].duplicated().any():
        errors.append("image_path 중복")
    return errors


def test_export_passes_model1_checks(dummy):
    _, rows = dummy
    result, _ = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    out = split.to_model1(result, variants=["original", "kakao"])
    assert list(out.columns) == split.MODEL1_COLUMNS
    assert len(out) == 2 * len(result)
    assert _model1_errors(out) == []
    row = out[(out["postprocess"] == "kakao") & (out["image_path"].str.contains("fakeA_jaehwan_mouse_001_s1"))].iloc[0]
    assert row["image_path"] == "data/processed/kakao/fakeA_jaehwan_mouse_001_s1.jpg"
    assert row["source_id"] == "jaehwan_mouse_001" and row["item_id"] == "mouse" and row["domain"] == "trade"


def test_export_paths_point_to_postprocess_outputs(dummy):
    root, rows = dummy
    data = root / "data"
    postprocess.run(data, CFG, ["original"])
    result, _ = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    out, missing = split.drop_missing(split.to_model1(result, ["original", "recapture"]), data)
    # original은 전부 있고, recapture는 사람이 찍은 일부만 있어서 나머지는 빠진다
    assert (out["postprocess"] == "original").sum() == len(result)
    assert set(missing["postprocess"]) == {"recapture"}


def test_export_rejects_unknown_variant(dummy):
    _, rows = dummy
    result, _ = split.build(rows, [0.7, 0.15, 0.15], seed=42)
    with pytest.raises(ValueError, match="후처리"):
        split.to_model1(result, ["jpeg50"])


# ---- metadata (파일명·EXIF 자동 생성) ----

def test_metadata_reads_filename_and_exif(dummy):
    root, _ = dummy
    meta = pd.read_csv(root / "metadata" / "real_metadata.csv", dtype=str, keep_default_na=False)
    assert list(meta.columns) == metadata.REAL_METADATA_COLUMNS
    row = meta.set_index("filename").loc["taei_keyboard_002.jpg"]
    assert (row["photographer"], row["object"], row["memo"], row["phone_model"]) == ("taei", "keyboard", "1", "DummyPhone")
    assert (row["width"], row["height"], row["taken_at"]) == ("640", "480", "2026-11-17 10:02:00")
    row = meta.set_index("filename").loc["taei_keyboard_nomemo_001.jpg"]
    assert (row["photographer"], row["object"], row["memo"]) == ("taei", "keyboard", "0")


def test_metadata_reports_bad_names_and_missing_exif(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    exif = Image.Exif()
    exif[0x0110] = "iPhone 15"
    exif[0x0112] = 6  # 세로로 찍어 90도 회전 정보만 붙은 사진
    Image.new("RGB", (40, 30)).save(real / "taei_mouse_001.jpg", exif=exif)
    Image.new("RGB", (40, 30)).save(real / "taei_mouse_002.jpg")  # 카톡 전송본처럼 EXIF 없음
    Image.new("RGB", (40, 30)).save(real / "IMG_1234.jpg")
    Image.new("RGB", (40, 30)).save(real / "taei_fridgee_001.jpg")
    meta, errors, warnings = metadata.build(tmp_path, objects={"mouse", "fridge"})
    assert list(meta["filename"]) == ["taei_mouse_001.jpg", "taei_mouse_002.jpg"]
    assert (meta.loc[0, "width"], meta.loc[0, "height"]) == (30, 40)
    assert len(errors) == 2 and errors[0].startswith("IMG_1234.jpg") and "objects.csv" in errors[1]
    assert len(warnings) == 1 and warnings[0].startswith("taei_mouse_002.jpg")


def test_objects_csv_matches_plan():
    objects = pd.read_csv("metadata/objects.csv", dtype=str)
    # 이슈 #5: 24종 중 20종 확정. 추가 4종은 근영님이 정하면 넣고 24로 바꾼다
    assert list(objects.columns) == ["object", "name_ko"] and len(objects) == 20
    assert objects["object"].str.fullmatch(r"[a-z0-9]+").all() and objects["object"].is_unique
    assert "car" in set(objects["object"]) and "bicycle" not in set(objects["object"])
    assert schema.NOMEMO not in set(objects["object"])  # 물건 이름이 nomemo면 파일명을 구분할 수 없음


# ---- 파일명 규칙 (메모 없는 사진) ----

def test_real_name_cleans_drive_names():
    assert schema.real_name("giwon_smartwatch_014 - 차기원.jpg") == "giwon_smartwatch_014.jpg"
    assert schema.real_name("Yoojaehwan_headphone_001.JPG") == "yoojaehwan_headphone_001.jpg"
    assert schema.real_name("taei_mouse_nomemo_003.jpg") == "taei_mouse_nomemo_003.jpg"


def test_nomemo_names_parse_for_real_and_fakes():
    m = schema.REAL_NAME.match("taei_mouse_nomemo_003.jpg")
    assert (m["photographer"], m["object"], m["nomemo"], m["idx"]) == ("taei", "mouse", "nomemo", "003")
    assert schema.REAL_NAME.match("taei_mouse_003.jpg")["nomemo"] is None
    assert not schema.REAL_NAME.match("taei_mouse_memo_003.jpg")

    fake_b = schema.fake_ab_name("B", "taei_mouse_nomemo_003.jpg", seed=42)
    assert fake_b == "fakeB_taei_mouse_nomemo_003_s42.png"
    assert schema.FAKE_AB_NAME.match(fake_b)["source"] == "taei_mouse_nomemo_003"
    assert schema.group_id(fake_b) == "taei_mouse_nomemo_003"

    fake_c = schema.fake_c_name("mouse", 3, seed=7, memo=False)
    assert fake_c == "fakeC_mouse_nomemo_003_s7.png"
    m = schema.FAKE_C_NAME.match(fake_c)
    assert (m["object"], m["nomemo"], m["idx"]) == ("mouse", "nomemo", "003")
    assert schema.fake_c_name("mouse", 3, seed=7) == "fakeC_mouse_003_s7.png"

    for name in ["taei_mouse_001.jpg", "fakeA_taei_mouse_001_s1.png", "fakeC_mouse_001_s1.png"]:
        assert schema.has_memo(name)
    for name in ["taei_mouse_nomemo_001.jpg", fake_b, fake_c]:
        assert not schema.has_memo(name)
