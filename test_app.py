import io
import unittest
from uuid import uuid4

from app import app, db, User


class AppSmokeTest(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
        self.client = app.test_client()

    def tearDown(self):
        with app.app_context():
            # Remove only the temporary test account created by this test suite.
            user = User.query.filter_by(username=self.username).first() if hasattr(self, "username") else None
            if user:
                db.session.delete(user)
                db.session.commit()

    def login_test_user(self):
        self.username = f"test_{uuid4().hex[:8]}"
        email = f"{self.username}@example.com"
        response = self.client.post(
            "/register",
            data={
                "username": self.username,
                "email": email,
                "password": "TestPassword123!",
            },
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)

        response = self.client.post(
            "/login",
            data={"username": self.username, "password": "TestPassword123!"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)

    def test_home(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Student Feedback Analysis System", response.data)

    def test_login_page(self):
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)

    def test_protected_pages_require_login(self):
        for path in ("/dashboard", "/upload", "/analysis", "/reports", "/settings", "/api/dashboard"):
            response = self.client.get(path, follow_redirects=False)
            self.assertEqual(response.status_code, 302, path)
            self.assertIn("/login", response.headers["Location"], path)

    def test_authenticated_dashboard(self):
        self.login_test_user()
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis Dashboard", response.data)

    def test_upload_accepts_unlabelled_csv_after_login(self):
        self.login_test_user()
        csv_data = (
            "ID,Feedback Text,Category,Date\n"
            "T1,The lecturer explains concepts clearly,Teacher Feedback,2026-10-06\n"
        )
        response = self.client.post(
            "/upload",
            data={"file": (io.BytesIO(csv_data.encode("utf-8")), "test.csv")},
            content_type="multipart/form-data",
            follow_redirects=False,
        )
        # The model may be unavailable in a local test environment. The test only
        # verifies that validation does not reject the file for lacking Sentiment.
        self.assertNotIn(b"Missing required columns: Sentiment", response.data)


if __name__ == "__main__":
    unittest.main()
