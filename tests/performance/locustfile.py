from locust import HttpUser, task, between

class StreamlitUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    @task
    def visit_homepage(self):
        self.client.get("/")
    
    @task(2)  # This task runs twice as often
    def interact_with_app(self):
        # Add your app-specific interactions here
        # Example:
        # self.client.post("/process_text", json={"text": "Sample text"})
        pass 