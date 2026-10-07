import io
import unittest
from app import app

class AppSmokeTest(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_dashboard(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis Dashboard", response.data)

    def test_upload_page(self):
        response = self.client.get("/upload")
        self.assertEqual(response.status_code, 200)

    def test_upload_unlabelled_csv(self):
        csv_data = (
            "ID,Feedback Text,Category,Date\n"
            "T1,The lecturer explains concepts clearly,Teacher Feedback,2026-10-06\n"
        )
        with app.test_client() as client:
            response = client.post(
                "/upload",
                data={"file": (io.BytesIO(csv_data.encode("utf-8")), "test.csv")},
                content_type="multipart/form-data",
                follow_redirects=False,
            )
        # Either a successful analysis redirect or a model-related redirect is acceptable
        # for environments where the trained model is not present. The old validation
        # error for a missing Sentiment column must not occur.
        self.assertNotIn(b"Missing required columns: Sentiment", response.data)

    def test_analysis(self):
        response = self.client.get("/analysis")
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
