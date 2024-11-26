from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# Prompt v1
# prmpt_string = """
# You're an expert OCR text cleaner.  You look at text produced by an OCR model and then you review that 
# text and perform the following:

# 1 - Correct misspellings
# 2 - Reorganize data coming from forms so it's more readable
# 3 - Reorganize data coming from tables so it's more readable

# Here is an example of what form text might look like:

# NAMP
# DATE
# CITY
# STATE ZIP
# 8-3-89

# Here is an example of what form text should be revised to look like:

# NAME:
# DATE: 8-3-89
# CITY: 
# STATE: 
# ZIP:


# Please only return a revised version of the original text.  

# Here is the text to revise:
# """

# Prompt v2 - smaller, misspelling correction only
prmpt_string = """
You're an expert OCR text cleaner.  You look at text produced by an OCR model and then you review that 
text and correct misspellings.

Please revise all misspellings.  Only revise spelling.  Don't revise anything else.  Only 
correct spelling where you are absolutely sure the word is misspelled and you're absolutely
sure what the correct spelling is.  Please return the original text with misspelling corrections applied. 

Here is the text to revise:
"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            prmpt_string,
        ),
        # MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)

# llm = ChatOllama(model="gemma2")
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)

reviser_chain = prompt | llm