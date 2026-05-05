import json
from unittest.mock import patch, MagicMock
import pytest
import requests
from PIL import Image

from extractor import ModelResponse, TitleBlockExtractor


# - - - - - - - - - - DATA MODELS - - - - - - - - - -

class TestCoerceToList:
    def test_string_becomes_list(self):
        r = ModelResponse(scale="1:50")
        assert r.scale == ["1:50"]

    def test_empty_list_becomes_none(self):
        r = ModelResponse(scale=[])
        assert r.scale is None

    def test_list_passes_through(self):
        r = ModelResponse(date=["01.01.2020", "02.02.2021"])
        assert r.date == ["01.01.2020", "02.02.2021"]

    def test_none_stays_none(self):
        r = ModelResponse(bkp_code=None)
        assert r.bkp_code is None


class TestScaleValidation:
    @pytest.mark.parametrize("raw, expected", [
        ("1:50", ["1:50"]),
        ("1:100", ["1:100"]),
        ("1:1,000", ["1:1,000"]),
        ("1:1.000", ["1:1.000"]),
    ])
    def test_valid_scales(self, raw, expected):
        assert ModelResponse(scale=raw).scale == expected

    def test_spaces_around_colon_normalized(self):
        assert ModelResponse(scale="1 : 50").scale == ["1:50"]

    def test_multiple_scales(self):
        r = ModelResponse(scale=["1:50", "1:100"])
        assert r.scale == ["1:50", "1:100"]

    @pytest.mark.parametrize("raw", [
        "50",
        "1/50",
        "one to fifty",
        "1:abc",
    ])
    def test_invalid_scales_become_none(self, raw):
        assert ModelResponse(scale=raw).scale is None

    def test_mix_valid_and_invalid(self):
        r = ModelResponse(scale=["1:50", "bad", "1:100"])
        assert r.scale == ["1:50", "1:100"]

    def test_all_invalid_becomes_none(self):
        r = ModelResponse(scale=["bad", "worse"])
        assert r.scale is None


class TestDateValidation:
    @pytest.mark.parametrize("raw, expected", [
        ("01.05.2020", ["01.05.2020"]),
        ("31.12.1999", ["31.12.1999"]),
        ("03.2025", ["03.2025"]),
    ])
    def test_valid_dates(self, raw, expected):
        assert ModelResponse(date=raw).date == expected

    def test_two_digit_year_recent(self):
        assert ModelResponse(date="15.03.24").date == ["15.03.2024"]

    def test_two_digit_year_old(self):
        assert ModelResponse(date="09.09.81").date == ["09.09.1981"]

    def test_two_digit_year_boundary(self):
        assert ModelResponse(date="01.01.30").date == ["01.01.2030"]
        assert ModelResponse(date="01.01.31").date == ["01.01.1931"]

    def test_month_year_two_digit(self):
        assert ModelResponse(date="03.24").date == ["03.2024"]

    def test_transposed_year_month(self):
        assert ModelResponse(date="2025.03").date == ["03.2025"]

    @pytest.mark.parametrize("raw", [
        "2020-01-05",
        "January 2020",
        "abc",
    ])
    def test_invalid_dates_become_none(self, raw):
        assert ModelResponse(date=raw).date is None

    def test_mix_valid_and_invalid(self):
        r = ModelResponse(date=["01.05.2020", "not-a-date", "03.2025"])
        assert r.date == ["01.05.2020", "03.2025"]

    def test_multiple_dates(self):
        r = ModelResponse(date=["09.09.1981", "17.09.1981"])
        assert r.date == ["09.09.1981", "17.09.1981"]


class TestBkpCodeValidation:
    @pytest.mark.parametrize("raw, expected", [
        ("2", ["2"]),
        ("25", ["25"]),
        ("256", ["256"]),
        ("2561", ["2561"]),
        ("256.2", ["256.2"]),
    ])
    def test_valid_codes(self, raw, expected):
        assert ModelResponse(bkp_code=raw).bkp_code == expected

    def test_multiple_codes(self):
        r = ModelResponse(bkp_code=["214", "221.1"])
        assert r.bkp_code == ["214", "221.1"]

    @pytest.mark.parametrize("raw", [
        "12345",
        "BKP 221",
        "abc",
        "22.1",
        "1234.5",
    ])
    def test_invalid_codes_become_none(self, raw):
        assert ModelResponse(bkp_code=raw).bkp_code is None

    def test_mix_valid_and_invalid(self):
        r = ModelResponse(bkp_code=["221", "invalid", "256.2"])
        assert r.bkp_code == ["221", "256.2"]


class TestStringFields:
    def test_plan_no_passthrough(self):
        r = ModelResponse(plan_no="0324_A_214_4015")
        assert r.plan_no == "0324_A_214_4015"

    def test_plan_name_passthrough(self):
        r = ModelResponse(plan_name="Südfassade Detail")
        assert r.plan_name == "Südfassade Detail"

    def test_all_none_by_default(self):
        r = ModelResponse()
        assert r.plan_no is None
        assert r.plan_name is None
        assert r.scale is None
        assert r.date is None
        assert r.bkp_code is None


# - - - - - - - - - - FIXTURES - - - - - - - - - -

@pytest.fixture
def extractor():
    return TitleBlockExtractor()

@pytest.fixture
def dummy_image():
    return Image.new("RGB", (100, 100), color="white")

def _ollama_response(body: dict) -> MagicMock:
    mock = MagicMock()
    mock.json.return_value = {"response": json.dumps(body)}
    mock.raise_for_status = MagicMock()
    return mock

FULL_RESPONSE = {
    "plan_no": "A-101",
    "plan_name": "Grundriss EG",
    "scale": ["1:50"],
    "date": ["01.05.2020"],
    "bkp_code": ["221"],
}

EMPTY_RESPONSE = {
    "plan_no": None,
    "plan_name": None,
    "scale": None,
    "date": None,
    "bkp_code": None,
}


# - - - - - - - - - - _infer() - - - - - - - - - -

class TestInfer:
    @patch("extractor.requests.post")
    def test_valid_json(self, mock_post, extractor):
        mock_post.return_value = _ollama_response(FULL_RESPONSE)
        notes = []
        result = extractor._infer("b64img", "prompt", notes)
        assert result.plan_no == "A-101"
        assert result.scale == ["1:50"]
        assert result.date == ["01.05.2020"]
        assert notes == []

    @patch("extractor.requests.post")
    def test_malformed_json(self, mock_post, extractor):
        mock = MagicMock()
        mock.json.return_value = {"response": "not json at all"}
        mock.raise_for_status = MagicMock()
        mock_post.return_value = mock
        notes = []
        result = extractor._infer("b64img", "prompt", notes)
        assert result is None
        assert any("malformed JSON" in n for n in notes)

    @patch("extractor.requests.post")
    def test_non_object_json(self, mock_post, extractor):
        mock = MagicMock()
        mock.json.return_value = {"response": json.dumps([1, 2, 3])}
        mock.raise_for_status = MagicMock()
        mock_post.return_value = mock
        notes = []
        result = extractor._infer("b64img", "prompt", notes)
        assert result is None
        assert any("non-object JSON" in n for n in notes)

    @patch("extractor.requests.post")
    def test_missing_keys_filled_with_none(self, mock_post, extractor):
        mock_post.return_value = _ollama_response({"plan_no": "X-1"})
        notes = []
        result = extractor._infer("b64img", "prompt", notes)
        assert result.plan_no == "X-1"
        assert result.plan_name is None
        assert result.scale is None

    @patch("extractor.requests.post")
    def test_markdown_fences_stripped(self, mock_post, extractor):
        mock = MagicMock()
        mock.json.return_value = {
            "response": '```json\n{"plan_no": "Z-9", "plan_name": null, "scale": null, "date": null, "bkp_code": null}\n```'
        }
        mock.raise_for_status = MagicMock()
        mock_post.return_value = mock
        notes = []
        result = extractor._infer("b64img", "prompt", notes)
        assert result.plan_no == "Z-9"

    @patch("extractor.requests.post")
    def test_connection_error(self, mock_post, extractor):
        mock_post.side_effect = requests.ConnectionError()
        notes = []
        result = extractor._infer("b64img", "prompt", notes)
        assert result is None
        assert any("could not connect" in n for n in notes)

    @patch("extractor.requests.post")
    def test_timeout(self, mock_post, extractor):
        mock_post.side_effect = requests.Timeout()
        notes = []
        result = extractor._infer("b64img", "prompt", notes)
        assert result is None
        assert any("timed out" in n for n in notes)

    @patch("extractor.requests.post")
    def test_validation_failure_appends_note(self, mock_post, extractor):
        body = {**EMPTY_RESPONSE, "scale": "bad_scale"}
        mock_post.return_value = _ollama_response(body)
        notes = []
        result = extractor._infer("b64img", "prompt", notes)
        assert result.scale is None
        assert any("scale failed validation" in n for n in notes)


# - - - - - - - - - - extract() - - - - - - - - - -

class TestExtract:
    @patch.object(TitleBlockExtractor, "_infer")
    def test_crop_succeeds_no_fallback(self, mock_infer, extractor, dummy_image):
        mock_infer.return_value = ModelResponse(**FULL_RESPONSE)
        result = extractor.extract(crop=dummy_image, thumbnail=dummy_image)
        assert result["plan_no"] == "A-101"
        assert result["scale"] == ["1:50"]
        assert result["fallback"] is False
        mock_infer.assert_called_once()

    @patch.object(TitleBlockExtractor, "_infer")
    def test_crop_empty_fallback_fires(self, mock_infer, extractor, dummy_image):
        crop_result = ModelResponse(**EMPTY_RESPONSE)
        fallback_result = ModelResponse(**FULL_RESPONSE)
        mock_infer.side_effect = [crop_result, fallback_result]
        result = extractor.extract(crop=dummy_image, thumbnail=dummy_image)
        assert result["plan_name"] == "Grundriss EG"
        assert result["date"] == ["01.05.2020"]
        assert result["fallback"] is True
        assert mock_infer.call_count == 2

    @patch.object(TitleBlockExtractor, "_infer")
    def test_fallback_merges_without_overwriting(self, mock_infer, extractor, dummy_image):
        crop_result = ModelResponse(plan_no="CROP-1", plan_name=None, scale=None, date=None, bkp_code=None)
        fallback_result = ModelResponse(plan_no="FB-1", plan_name="From Fallback", scale=["1:100"], date=["01.01.2025"], bkp_code=None)
        mock_infer.side_effect = [crop_result, fallback_result]
        result = extractor.extract(crop=dummy_image, thumbnail=dummy_image)
        assert result["plan_no"] == "CROP-1"
        assert result["plan_name"] == "From Fallback"
        assert result["scale"] == ["1:100"]
        assert result["fallback"] is True

    @patch.object(TitleBlockExtractor, "_infer")
    def test_crop_empty_no_thumbnail(self, mock_infer, extractor, dummy_image):
        mock_infer.return_value = ModelResponse(**EMPTY_RESPONSE)
        result = extractor.extract(crop=dummy_image, thumbnail=None)
        assert result["plan_no"] is None
        assert result["plan_name"] is None
        assert result["fallback"] is False

    @patch.object(TitleBlockExtractor, "_infer")
    def test_crop_none_fallback_none(self, mock_infer, extractor, dummy_image):
        mock_infer.side_effect = [None, None]
        result = extractor.extract(crop=dummy_image, thumbnail=dummy_image)
        assert result["plan_no"] is None
        assert result["fallback"] is False
        assert any("fallback inference returned" in n for n in result["notes"])

    @patch.object(TitleBlockExtractor, "_infer")
    def test_fallback_disabled(self, mock_infer, dummy_image):
        ext = TitleBlockExtractor(fallback=False)
        mock_infer.return_value = ModelResponse(**EMPTY_RESPONSE)
        result = ext.extract(crop=dummy_image, thumbnail=dummy_image)
        assert result["fallback"] is False
        mock_infer.assert_called_once()
