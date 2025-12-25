from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

class Assistant:
    def __init__(
        self,
        system_prompt,
        llm,
        message_history=[],
        vector_store=None,
        user_information=None,
    ):
        self.system_prompt = system_prompt
        self.llm = llm
        self.messages = message_history or []
        self.vector_store = vector_store
        self.user_information = user_information
        self.chain = self._get_conversation_chain()

    def get_response(self, user_query):
        return self.chain.invoke({
            "user_input": user_query,
        })

    def _get_conversation_chain(self):
        prompt = ChatPromptTemplate(
            messages=[
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="conversation_history"),
                ("user", "{user_input}"),
            ]
        )

        llm = self.llm
        output_parser = StrOutputParser()
        chain = (
            {
                "retrieved_policy_information": itemgetter("user_input") | self.vector_store.as_retriever(),
                "user_information": lambda x: self.user_information,
                "user_input": RunnablePassthrough(),
                "conversation_history": lambda x: self.messages,
            }
            | prompt
            | llm
            | output_parser
        )
        return chain
