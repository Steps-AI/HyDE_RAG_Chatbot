# Importing the streamlit library, which is used for creating web applications
import streamlit as st
# Importing the openai library, which allows for interactions with the OpenAI API, particularly useful for GPT models
import openai
import asyncio
import aiohttp
from lxml import html
from googlesearch import search


# Setting the API key for OpenAI. This is crucial for authenticating requests sent to the OpenAI API.
openai.api_key = ""

cache = {}


async def fetch(session, url, timeout=10):
    try:
        async with session.get(url, timeout=timeout) as response:
            return await response.text()
    except asyncio.TimeoutError:
        print(f"Timeout while fetching {url}")
        return None
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.ensure_future(fetch(session, url)) for url in urls]
        return await asyncio.gather(*tasks), urls

def clean_text(text_list):
    # Remove whitespace and line breaks, then filter out empty strings
    return [text.strip() for text in text_list if text.strip()]

def extract_data(raw_html):
    tree = html.fromstring(raw_html)

    data = {
        'headlines': clean_text(tree.xpath('//h1/text()')),
        'sub_headings': clean_text(tree.xpath('//h2/text() | //h3/text() | //h4/text() | //h5/text() | //h6/text()')),
        'paragraphs': clean_text(tree.xpath('//p/text()')),
        'lists': [],
        'tables': [],
    }

    # Extract lists
    list_elements = tree.xpath('//ul | //ol')
    for list_element in list_elements:
        items = clean_text(list_element.xpath('.//li/text()'))
        if items:  # Only add the list if it's not empty
            data['lists'].append(items)

    # Extract tables
    tables = tree.xpath('//table')
    for table in tables:
        headers = clean_text(table.xpath('.//th/text()'))
        rows = []
        for tr in table.xpath('.//tr'):
            cells = clean_text(tr.xpath('.//td/text() | .//td//a/text()'))  # Include hyperlink text
            if cells:  # Only add the row if it's not empty
                rows.append(cells)
        if headers or rows:  # Only add the table if it has content
            data['tables'].append({'headers': headers, 'rows': rows})

    return data

def web_data_retrival(query, num_results=2):
    # Check cache first
    if query in cache:
        return cache[query]

    # Get top URLs from Google search
    urls = [url for url in search(query, num_results=num_results)]

    # Fetch all URLs asynchronously using the existing event loop
    try:
        # Try to get the existing event loop
        loop = asyncio.get_event_loop()
    except RuntimeError as ex:
        # If no event loop is available in this thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        html_contents, urls = loop.run_until_complete(fetch_all(urls))
    except:
        html_contents = []

    # Extract and process data from each page
    results = {}
    
    try:
        for url, content in zip(urls, html_contents):
            if content:
                data = extract_data(content)
                results[url] = data
    except:
        pass

    # Save to cache
    cache[query] = results
    return results

# Defining a class named 'Chatbot', which encapsulates all the functionalities of the chatbot in the application.
class Chatbot:
    # Constructor method for the Chatbot class. It initializes the class with necessary attributes.
    def __init__(self):
        self.messages = []
        self.initialize_app_instance()
    
    # Initializes the app state and sets up initial conversation
    def initialize_app_instance(self):
        # Initialize session state for messages and model
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "system", "content": "You are a helpful AI QA assistant. "
                    "When answering questions, use the context enclosed by triple backquotes if it is relevant. "
                    "If you don't know the answer, just say that you don't know, "
                    "don't try to make up an answer. "
                    "Reply your answer in mardkown format."}) # -------> write your system mesage here !!!
            response_initial = openai.ChatCompletion.create(
            model= "gpt-3.5-turbo-1106",
            #response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a helpful AI QA assistant. "
                    "When answering questions, use the context enclosed by triple backquotes if it is relevant. "
                    "If you don't know the answer, just say that you don't know, "
                    "don't try to make up an answer. "
                    "Reply your answer in mardkown format."}, # -------> write your system mesage here !!!
                {"role": "user", "content": "Hi !"}
            ]
            )
            #print(response_initial.choices[0])
            # get the initial greeting message
            st.session_state.messages.append({"role": "assistant", "content": response_initial.choices[0]["message"]["content"]})
        
        # initialize session variables
        if "model" not in st.session_state:
            st.session_state.model =  "gpt-3.5-turbo-1106"
        if 'disabled' not in st.session_state:
            st.session_state.disabled = False
    
    
    
    # Displays greeting messages in the chat
    def dispay_greeting_message(self):
        # Iterate through user and system messages to display the greeting message
        for message in st.session_state["messages"]:
            if (message["role"] == "user") or (message["role"] == "assistant"):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])



# Main function to run the Streamlit app    
def main():
    # Setting the title of the Streamlit web app
    st.title("Steps AI Chatbot")
    # Instantiating the Chatbot class
    chatbot = Chatbot()
    # Initializing the app instance - setting up session states, initial messages
    chatbot.initialize_app_instance()
    # Displaying the initial greeting message in the chat interface
    chatbot.dispay_greeting_message()
    
    
    # Capturing user input from the chat interface
    user_prompt = st.chat_input("Your prompt",disabled=st.session_state.disabled)
    # Displaying the current state of tags (stores in the "return_tags" session variable) in the sidebar
    st.sidebar.subheader(st.session_state.return_tags)
    if user_prompt:
        user_prompt = user_prompt.lower()
        if (len(user_prompt.split()) >= 4) and ("yes" not in user_prompt) and ("no" not in user_prompt):
            temp_web_retrival_data = "The following is the web scrapped content based on the users' query/asnwer, refer to this infomration for more related infomration or to get latest updates from web. The scrapped data will be provide to you in this format ({url: {web scrapped data based on html components}, url: {webdata},..}) Please include citations (weblinks, article names, website-names etc.) in answers if you are refereing from the web scrapped content, the corresponding web links for each web scrapped data will be provided to you, if the scrapped content dosen't make any sense to you please ignore the web scrapped data, this data is provided just for your additional infomration so that you can get more latest infomration from the web \n \n" +" Web scrapped data based on the user prompt is : \n \n"+ str(web_data_retrival(user_prompt)) + "\n \n"
        else:
            temp_web_retrival_data = ""
        # Temporarily disabling the chat input to prevent new input while processing
        st.session_state.disabled = True
        st.chat_input("The answer is being generated, please wait...", disabled=st.session_state.disabled)
        # Appending the user's message to the session state
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        # Displaying the user's message in the chat interface
        with st.chat_message("user"):
            st.markdown(user_prompt)
    
        # Handling the assistant's response generation
        with st.chat_message("assistant"):
            # Placeholder for dynamic response display
            message_placeholder = st.empty()
            # Variable to accumulate the full response
            full_response = ""
            # Generating responses from OpenAI's API
            messages_temp=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            if temp_web_retrival_data == "":
                messages_temp.append({"role": "user", "content": user_prompt})
            else:
                messages_temp.append({"role": "user", "content": str(temp_web_retrival_data) + "based on the provided web scrapped data and the user prompt (i.e. input/query/answer) please respond accordingly.  The input User prompt is : " + user_prompt})
            #messages_temp.append({"role": "user", "content": str(temp_web_retrival_data) + "based on the provided web scrapped data and the user prompt (i.e. input/query/answer) please respond accordingly. Please include citations (weblinks, article names, website-names etc.) in answers if you are refereing from the web scrapped content, the corresponding web links for each web scrapped data will be provided to you, the scrapped data will be provide to you in this format ({url: {web scrapped data based on html components}, url: {webdata},..}). Please ignore the web scrapped data if it is irrelevant and dosen't make any sense to you. \n \n The input User prompt is : " + user_prompt})
            print(messages_temp)
            for response in openai.ChatCompletion.create(
                model=st.session_state.model,
                messages=messages_temp,
                stream=True,
            ):
                # Concatenating received response fragments
                full_response += response.choices[0].delta.get("content", "")
                # Dynamically updating the response in the chat UI placeholder
                message_placeholder.markdown(full_response + "▌")
                
            # Finalizing the display of the full response
            message_placeholder.markdown(full_response)
            #st.chat_input("Your prompt",disabled=True)
            
        # Appending the assistant's full response to the session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        # Re-enabling the chat input for further user interaction
        st.session_state.disabled = False
        # Re-running the Streamlit app to reflect updates
        st.rerun()


# Entry point for the script
if __name__ == "__main__":
    main()