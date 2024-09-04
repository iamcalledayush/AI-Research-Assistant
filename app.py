import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up Google API Key directly in the code
GOOGLE_API_KEY = "AIzaSyBoX4UUHV5FO4lvYwdkSz6R5nlxLadTHnU"

# Initialize the Gemini 1.5 Pro model
def init_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=api_key,
        temperature=0.8,  # Adjust temperature for creative responses
        max_tokens=300,    # Adjust max tokens for better performance
        timeout=15,        # Timeout after 15 seconds
        max_retries=3      # Retry up to 3 times in case of errors
    )

# Define the game’s story templates and logic
story_template = """
You are the main character in an epic journey. Your adventure begins in a {setting}, where you encounter {conflict}. 
Your goal is to {goal}, but along the way, you face challenges that test your {skills}. 
The choices you make will determine the outcome of your story.

Here are the starting details:
Setting: {setting}
Conflict: {conflict}
Goal: {goal}

What would you like to do first?
"""

# Define PromptTemplates for dynamically generating the story
story_prompt = PromptTemplate(
    input_variables=["setting", "conflict", "goal", "skills"],
    template=story_template
)

# Create LLMChain for the story
def create_llm_chain(llm, prompt):
    return LLMChain(
        llm=llm,
        prompt=prompt
    )

# Streamlit interface for the game
st.title("Interactive Fiction Game with Dynamic Storylines")

# Initialize session state variables to maintain game flow
if "story_output" not in st.session_state:
    st.session_state.story_output = ""

st.write("### Welcome to your personalized adventure! 🎮")
st.write("Your journey will adapt to the choices you make, creating unique storylines along the way.")

# Input fields for the player to create their own adventure elements
setting = st.text_input("Enter the setting for your story (e.g., a mystical forest, a futuristic city):", "a mystical forest")
conflict = st.text_input("Enter the conflict you face (e.g., a dragon terrorizing the land, an evil corporation plotting destruction):", "a dragon terrorizing the land")
goal = st.text_input("Enter your main goal (e.g., retrieve the ancient artifact, save the world):", "retrieve the ancient artifact")
skills = st.text_input("Enter the skills your character possesses (e.g., swordsmanship, magic, detective skills):", "magic")

# Button to start the game
if st.button("Start Your Adventure"):
    with st.spinner("Weaving your story..."):
        llm = init_llm(GOOGLE_API_KEY)  # Use the primary API key for the LLM
        story_chain = create_llm_chain(llm, story_prompt)
        # Generate the first part of the story
        st.session_state.story_output = story_chain.run({
            "setting": setting,
            "conflict": conflict,
            "goal": goal,
            "skills": skills
        })

# Display the story output
if st.session_state.story_output:
    st.write("### Your Story Begins:")
    st.write(st.session_state.story_output)

    # Provide the player with a choice for what to do next
    next_action = st.text_input("What would you like to do next in the story?")
    
    if st.button("Continue"):
        with st.spinner("Continuing your journey..."):
            # Update the story based on the player's next action
            try:
                llm = init_llm(GOOGLE_API_KEY)  # Use the same API key for continuity
                next_story_prompt = f"{st.session_state.story_output}\n\nPlayer's Action: {next_action}\nContinue the story based on this."
                response = llm.invoke(next_story_prompt)
                st.session_state.story_output += "\n\n" + response.content
                st.write(st.session_state.story_output)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}. Please try again.")
