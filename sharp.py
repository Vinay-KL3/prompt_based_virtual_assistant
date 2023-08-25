#I have to focous more on below 3 points, as i took help from chatgpt for those codes
#audio recorder for ASR
#streamlit session manager
#streamlit output displaying




import os
os.environ['OPENAI_API_KEY'] = "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"

def prompt_format(input_name, input_amount):
    dummy_name = "{Name}"
    # input_name = "Raahul"
    # input_amount = "2500"
    prompt = """
            You are a virtual chatbot agent chatting with a live customer and asking them to clear their EMI due payment immediately.
            task = you have to behave like a live chat bot, and start conversation with {0}, who has a pending EMI of {1} rupees to be paid. Convince {2} to do the payment.
            process = you just ask one short question and wait for the customer to respond. you can respond again according to customer's response.
            primary_question = ask customer, by when will he/she make the payment
            Endining_Response = Thank you for chatting with us, Have a great day!    
            important_parameter = keep all your responses short and crisp, not more than 120 charecters.
            Goal = educate and convince the customer to do the payment immediately.
            additional_pointers = continue the conversation until the customer agrees to pay or says bye, or thank you. end the final conversation with exact Endining_Response""".format(dummy_name, input_amount, input_name)

    #LangChain

    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate

    prompt_template_name = PromptTemplate(
        input_variables =['Name'],
        template = prompt
    )
    p = prompt_template_name.format(Name=input_name)
    return p
# print(p)



#-----------------------------------------------------------------------

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
    
def bot(intake):

    memory = ConversationBufferWindowMemory (k=7)

    convo = ConversationChain(
                                llm = OpenAI(temperature=0.6),
                                memory=memory
                            )
    BOT = convo.run(intake)
    print("\nBOT : ", BOT)
    return BOT
    




from gtts import gTTS
from playsound import playsound

def TTS(text):
    lang = "en"
    text = text

    path = "Aug22.mp3"

    speech = gTTS(text=text, lang=lang, slow=False, tld="com.au")
    speech.save(path)
    
    playsound(path)
    
    
    
    
import pyaudio
import wave

def ASR2():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []
    record_duration = 7
    iterations = int(44100/1024*record_duration)


    try:
         for _ in range(iterations):
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    audio.terminate()

    sound_file = wave.open("Aug23.mp3", "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(frames))
    sound_file.close()



    #Hugging Face Whisper ASR API

    import requests
    API_TOKEN = "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"
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(filename):
        with open(filename, "rb") as f:
            data = f.read()
        response = requests.post(API_URL, headers=headers, data=data)
        return response.json()

    output = query("Aug23.mp3")
    utterance = output["text"]
    print("USER : ", utterance)
    return utterance





# def bot(intake):
#     BOT = convo.run(intake)
#     print("\nBOT : ",BOT)
#     return BOT





#voice bot

def voice_bot(input_name, input_amount):
    p = prompt_format(input_name=input_name, input_amount=input_amount)
    BOT = bot(intake=p)
    # print("\nBOT : ",BOT)
    TTS(text=BOT)
    flag = True


    while flag == True:
        USER = ASR2()
        BOT = bot(intake=USER)
        TTS(text=BOT)
        if "thank you" in BOT or "Thank you" in BOT:
            flag = False























import streamlit as st




# Set page configuration
st.set_page_config(page_title="Sharp.Ai", page_icon="🔒")




# Page 1: Login Page
def login_page():
    # st.title("Login")
    
    st.image("/Users/vinay_kl/Downloads/promp_based_conversationl_bot/comb_agent_1.png", use_column_width=True)
    password = st.text_input("Please enter your sharp key below, and click login", type="password")
    if st.button("Login"):
        
        if password == "sharpUplisTorm1357":
            st.success("Login successful! Redirecting to the main page...")
            return True
        else:
            st.error("Seems like you entered a wrong password. Please re-check your password.")
    return False





# Page 2: Main Page
def main_page():
    col1, col2 = st.columns(2)
    
    col1.title("Welcome back!")
    
    sc1,sc2,sc3 = col2.columns(3)
    
    if sc2.button("Logout"):
        st.session_state.logged_in = False  # Reset the login state
        st.experimental_rerun()

    if sc3.button("Profile", help="profile feature is not implemented yet"):
        col2.write("Profile feature is not implemented yet") 
       
    # st.image("/Users/vinay_kl/Downloads/loading.png", use_column_width=True)    
    # Your content for the main page goes here
    
    # input_number = st.text_input("Enter Customer's Contact Number", type="tel")
    input_name = st.text_input("Enter Customer's Name")
    input_amount = st.text_input("Enter Pending EMI Amount")
    if st.button("Start Conversation"):
        st.write("**Conversation will start in next 5 seconds.**")
        st.text("Please note, there will be 5 sec delay for each conversation due to free sourced ASR")
        # st.write(input_name, input_amount)
        chat = voice_bot(input_name=input_name, input_amount=input_amount)







# Streamlit app
def main():
    st.sidebar.image("/Users/vinay_kl/Downloads/promp_based_conversationl_bot/logo png crop.png")
    st.sidebar.title("Navigation")
    
    if not hasattr(st.session_state, 'logged_in'):
        st.session_state.logged_in = False
    
    # Display login page by default
    page = "Login" if not st.session_state.logged_in else "Main"
    
    # Check if login is successful
    if login_page():
        st.session_state.logged_in = True
        page = "Main"

    # Show the appropriate page based on user input
    if page == "Login":
        st.sidebar.radio("Available Pages", ["Login", "Main"], index=0)  # Select "Login" by default
    elif page == "Main":
        main_page()
        st.sidebar.radio("Available Pages", ["Login", "Main"], index=1)  # Select "Main" by default


    st.sidebar.text("© 2023 Sharp.Ai")
    
    
    
    

if __name__ == "__main__":
    main()
