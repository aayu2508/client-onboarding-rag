import streamlit as st

class AssistantGUI:
    def __init__(self, assistant):

        # Get resources from assistant (mainly the response)
        self.assistant = assistant
        # Visualise the history of the chat
        self.messages = assistant.messages
        # Visualise the employee information
        self.user_information = assistant.user_information

    def get_response(self, user_input):
        return self.assistant.get_response(user_input)
    
    def render_messages(self):
        for message in self.messages:
            if message["role"] == "user":
                st.chat_message("human").markdown(message["content"])
            else:
                st.chat_message("ai").markdown(message["content"])
    
    def render_user_input(self):
        user_input = st.chat_input("Type here...", key="input")
        if not user_input:
            return
        
        self.messages.append({"role": "user", "content": user_input})
        st.session_state["messages"] = self.messages
        st.chat_message("human").markdown(user_input)

        response = self.get_response(user_input)
        self.messages.append({"role": "assistant", "content": response})
        st.session_state["messages"] = self.messages
        st.chat_message("assistant").markdown(response)

    def render(self):
        with st.sidebar:
            st.logo(
                "https://upload.wikimedia.org/wikipedia/commons/0/0e/Umbrella_Corporation_logo.svg"
            )
            st.title("Umbrella Corporation Assistant")
            st.subheader("Employee Information")
            st.write(self.user_information)

        self.render_messages()
        self.render_user_input()