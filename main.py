import pprint
from app import app

def run_interactive_mode():
    # Setup and compile the workflow from the imported function
    application = app()

    while True:
        # Ask for user input
        user_question = input("Please enter your question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            print("Exiting...")
            break

        # Prepare inputs for the workflow with the user's question
        inputs = {
            "keys": {
                "question": user_question,
                "local": "No",  # Or adjust based on whether you want to run locally or not
            },
        }

        # Run the workflow with the inputs
        for output in application.stream(inputs):
            for key, value in output.items():
                pprint.pprint(f"Node '{key}':")
            pprint.pprint("\n---\n")

        # Handle the final generation
        pprint.pprint(value['keys']['generation'])

if __name__ == "__main__":
    run_interactive_mode()