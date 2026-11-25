# This will test the environment to ensure that the .env file is set up 
# correctly and that the OpenAI and Neo4j connections are working.
import os
import unittest

from dotenv import load_dotenv, find_dotenv
load_dotenv()

def openai_error_hint(error):
    """A friendly next step for the common workshop LLM proxy failure classes."""
    status = getattr(error, "status_code", None)
    if status == 429:
        return "Your Workshop AI access panel key is out of quota - generate a new key in the panel or ask your facilitator."
    if status == 401:
        return "Your OPENAI_API_KEY was rejected - copy a fresh key from the Workshop AI access panel in the course."
    if status == 404:
        return "The model or URL was not found on the proxy - check the OPENAI_BASE_URL value in .env and that this model is enabled for your key."
    return "Check the OPENAI_API_KEY and OPENAI_BASE_URL values in .env file."


class TestEnvironment(unittest.TestCase):

    skip_env_variable_tests = True
    skip_openai_test = True
    skip_neo4j_test = True

    def test_env_file_exists(self):
        env_file_exists = True if find_dotenv() > "" else False
        if env_file_exists:
            TestEnvironment.skip_env_variable_tests = False
        self.assertTrue(env_file_exists, ".env file not found.")

    def env_variable_exists(self, variable_name):
        self.assertIsNotNone(
            os.getenv(variable_name),
            f"{variable_name} not found in .env file")

    def test_openai_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping OpenAI env variable test")

        self.env_variable_exists('OPENAI_API_KEY')
        self.env_variable_exists('OPENAI_BASE_URL')

        # A personal OpenAI key (sk-) will not work against the proxy and is
        # what people fall back to when the proxy key fails - fail early.
        key = os.getenv('OPENAI_API_KEY')
        self.assertFalse(
            key.startswith('sk-'),
            "OPENAI_API_KEY looks like a personal OpenAI key (sk-...). Use the ga-... key from the Workshop AI access panel in the course instead.")

        TestEnvironment.skip_openai_test = False

    def test_neo4j_variables(self):
        if TestEnvironment.skip_env_variable_tests:
            self.skipTest("Skipping Neo4j env variables test")

        self.env_variable_exists('NEO4J_URI')
        self.env_variable_exists('NEO4J_USERNAME')
        self.env_variable_exists('NEO4J_PASSWORD')
        self.env_variable_exists('NEO4J_DATABASE')
        TestEnvironment.skip_neo4j_test = False

    def test_openai_connection(self):
        if TestEnvironment.skip_openai_test:
            self.skipTest("Skipping OpenAI test")

        from openai import OpenAI

        # The OpenAI client reads OPENAI_API_KEY and OPENAI_BASE_URL from the
        # environment, so this call goes to the GraphAcademy workshop LLM proxy.
        llm = OpenAI()

        try:
            llm.embeddings.create(
                model="text-embedding-3-small",
                input="test"
            )
        except Exception as e:
            self.fail(
                f"OpenAI embeddings request failed: {type(e).__name__}: {e}\n{openai_error_hint(e)}")

        # The kg builders use gpt-5-nano and text2cypher/agent use gpt-5.2, so
        # validate the chat completions route at setup too, not just embeddings.
        try:
            llm.chat.completions.create(
                model="gpt-5-nano",
                messages=[{"role": "user", "content": "Say OK"}],
                max_completion_tokens=5
            )
        except Exception as e:
            self.fail(
                f"OpenAI chat completions request failed: {type(e).__name__}: {e}\n{openai_error_hint(e)}")

    def test_neo4j_connection(self):

        msg = "Neo4j connection failed. Check the NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4j_DATABASE values in .env file."
        connected = False

        if TestEnvironment.skip_neo4j_test:
            self.skipTest("Skipping Neo4j connection test")

        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            os.getenv('NEO4J_URI'),
            auth=(os.getenv('NEO4J_USERNAME'), 
                  os.getenv('NEO4J_PASSWORD'))
        )
        try:
            driver.verify_connectivity()
            try:
                driver.execute_query("RETURN true", database_=os.getenv('NEO4J_DATABASE'))
                connected = True

            except Exception as e:
                msg = "Neo4j database query failed. Check the NEO4J_DATABASE value in .env file."
                
        except Exception as e:
            msg = "Neo4j verify connection failed. Check the NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD values in .env file."
            
        driver.close()

        self.assertTrue(
            connected,
            msg
            )
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestEnvironment('test_env_file_exists'))
    suite.addTest(TestEnvironment('test_openai_variables'))
    suite.addTest(TestEnvironment('test_neo4j_variables'))
    suite.addTest(TestEnvironment('test_openai_connection'))
    suite.addTest(TestEnvironment('test_neo4j_connection'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    