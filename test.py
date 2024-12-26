## USE THIS FOR TESTING
import aisuite as ai
import os
import requests
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
from urllib.parse import quote
import logging
import json
from flask import Flask, request, render_template, Response
import sys

def format_sse_message(type_str, **kwargs):
    data = {"type": type_str}
    data.update(kwargs)
    return f"data: {json.dumps(data)}\n\n"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
config = {
    "semrush_database": "us",
    "semrush_display_limit": 50,
    "semrush_display_filter": "%2B%7CPo%7CLt%7C50",
    "semrush_display_sort": "po_asc",
    "jina_api_timeout": 15,
    "openai_model": "openai:gpt-4o-mini",
    "openai_temperature": 0.8
}

# --- Initialization ---
client = ai.Client()
load_dotenv()

# --- Helper Functions ---

def handle_api_errors(response, api_name):
    if response.status_code != 200:
        logging.error(f"Failed to retrieve data from {api_name}. Status code: {response.status_code}")
        return False
    return True

# --- Step 2: SerpAPI Data Retrieval ---

def get_serpapi_data(topic_query):
    base_url = "https://serpapi.com/search.json"
    params = {
        "q": topic_query,
        "hl": "en",
        "gl": "us",
        "google_domain": "google.com",
        "api_key": os.getenv("SERPAPI_KEY")
    }
    import time
    time.sleep(2)  # Simulate API call delay
    # Dummy data
    data = [
        {'Position': 1, 'Link': 'http://example.com/1', 'Title': 'Example Title 1'},
        {'Position': 2, 'Link': 'http://example.com/2', 'Title': 'Example Title 2'}
    ]
    df_serp = pd.DataFrame(data)
    logging.info("Using dummy SerpAPI data.")
    return df_serp

# --- Step 3: SEMRush Data Retrieval and Processing ---

def get_semrush_data(url, api_key=os.getenv("SEMRUSH_API_KEY")):
    base_url = "https://api.semrush.com/"
    type_param = "url_organic"
    export_columns = "Ph,Po,Nq,Cp,Co"
    full_url = (
        f"{base_url}?type={type_param}&key={api_key}"
        f"&display_limit={config['semrush_display_limit']}&export_columns={export_columns}"
        f"&url={quote(url)}&database={config['semrush_database']}"
        f"&display_filter={config['semrush_display_filter']}&display_sort={config['semrush_display_sort']}"
    )
    import time
    time.sleep(2)  # Simulate API call delay
    # Dummy data
    json_data = [
        {'Keyword': 'example', 'Search Volume': '1000', 'Position': '1'},
        {'Keyword': 'test', 'Search Volume': '500', 'Position': '2'}
    ]
    logging.info("Using dummy SEMRush data.")
    return json_data

def process_semrush_data(df_results):
    df_results['SEMRush_Data'] = df_results['Link'].apply(get_semrush_data)
    logging.info("Successfully retrieved SEMRush data.")
    all_keywords = []
    for data in df_results['SEMRush_Data']:
        if data:
            all_keywords.extend([item['Keyword'] for item in data])

    keyword_counts = Counter(all_keywords)
    highest_count = max(keyword_counts.values())
    second_highest_count = sorted(set(keyword_counts.values()), reverse=True)[1] if len(set(keyword_counts.values())) > 1 else 0
    top_keywords = [keyword for keyword, count in keyword_counts.items() if count == highest_count or count == second_highest_count]

    if highest_count == 2:
        top_keywords = [keyword for keyword, count in keyword_counts.items() if count in [1, 2]]

    search_volume_keywords = sorted(
        [(item['Keyword'], int(item['Search Volume'])) for data in df_results['SEMRush_Data'] if data for item in data],
        key=lambda x: x[1],
        reverse=True)[:10]

    final_keywords = set(top_keywords + [keyword for keyword, _ in search_volume_keywords])
    final_keywords_df = pd.DataFrame(
        [(keyword, 
        next((item['Search Volume'] for data in df_results['SEMRush_Data'] if data for item in data if item['Keyword'] == keyword), 0),
        keyword_counts[keyword])
        for keyword in final_keywords],
        columns=['Keyword', 'Search Volume', 'Frequency']
    )
    final_keywords_df = final_keywords_df.sort_values(by=['Frequency', 'Search Volume'], ascending=[False, False])

    logging.info("Successfully processed SEMRush data and extracted keywords.")
    return final_keywords_df

# --- Step 4: Content Fetching ---

def fetch_content(url):
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
        'X-Retain-Images': 'none',
        "Accept": "application/json",
        'X-Timeout': str(config["jina_api_timeout"])
    }
    try:
        import time
        time.sleep(2)  # Simulate API call delay
        # Dummy content
        logging.info(f"Using dummy content for {url}.")
        return "This is dummy content for testing purposes."

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error fetching content from {url}: {e}")
        return f"ERROR: Request failed for {url}."
    except Exception as e:
        logging.error(f"Unknown error fetching content from {url}: {e}")
        return f"ERROR: Unknown error processing {url}."

# --- Step 5-10: AI Model Interactions ---

def interact_with_ai(messages, model=config["openai_model"], temperature=config["openai_temperature"]):
    import time
    time.sleep(2)  # Simulate API call delay
    # Dummy response
    logging.info("Using dummy OpenAI response.")
    return "This is a dummy response for testing purposes."

# --- Main Workflow ---

def main():
    # Step 1: Select a Topic
    topic_query = input("Enter the topic you want to write about: ")  # Direct topic input
    
    # Step 2: Retrieve SERP Data
    df_serp = get_serpapi_data(topic_query)

    if df_serp.empty:
        print("Could not retrieve essential data. Exiting.")
        return

    # Step 3: Retrieve SEMrush Data
    df_results = df_serp.copy()
    final_keywords_df = process_semrush_data(df_results)

    # Step 4: Retrieve Onpage Content
    df_results['Content'] = df_results['Link'].apply(fetch_content)

    # Step 5: Content Analysis
    content_analysis = perform_content_analysis(df_results, topic_query)
    print(f"Content Analysis:\n{content_analysis}\n")

    # Step 6: Generate Content Plan
    content_plan = generate_content_plan(content_analysis, topic_query, final_keywords_df)
    print(f"Content Plan:\n{content_plan}\n")

    # Step 7: Generate Content Draft
    content_draft = create_content_draft(content_plan, content_analysis)
    print(f"Content Draft:\n{content_draft}\n")

    # Step 8: Proofread the Draft Post
    proofread_draft = proofread_content_draft(content_draft, content_plan, content_analysis)
    print(f"Proofread Draft:\n{proofread_draft}\n")

    # Step 9: SEO Recommendations
    seo_recommendations = provide_seo_recommendations(proofread_draft, final_keywords_df)
    print(f"SEO Recommendations:\n{seo_recommendations}\n")

    # Step 10: Final Deliverable
    final_deliverable = compile_final_deliverable(proofread_draft, seo_recommendations, final_keywords_df, df_serp, content_analysis)
    print(f"Final Deliverable:\n{final_deliverable}\n")

def perform_content_analysis(df_results, topic_query):
    content_list = [item['Content'] for item in df_results if item.get('Content') and not item['Content'].startswith('ERROR:')]
    messages_content_analysis = [
        {"role": "system", "content": "You are a meticulous content researcher with expertise in analyzing web content, particularly articles and blogs. You have access to a list of webpage contents related to the topic a user is interested in."},
        {"role": "user", "content": f"Analyze the provided content below. First, determine if each piece of content is a blog or an article. Disregard any content that is not a blog or an article. For each identified blog or article, add it to a review list. Then, thoroughly review each item on this list and provide an analysis that includes: (1) Common topics and subtopics covered across these blogs/articles. (2) Any contradicting viewpoints among the top 10 results. (3) For users searching for '{topic_query}', identify information gaps - what are they likely interested in that isn't covered, or what questions might they have that remain unanswered by these sources?\n\n"
                                + "\n".join([f"WEB CONTENT {i + 1}\n{content}" for i, content in enumerate(content_list)])
        }
    ]
    return interact_with_ai(messages_content_analysis)

def generate_content_plan(content_analysis, topic_query, final_keywords_df):
    messages_content_plan = [
        {"role": "system", "content": "You are an expert content strategist skilled in crafting detailed and actionable content plans. You are adept at creating outlines that are clear, comprehensive, and tailored to the specific needs of a given topic. You have access to a detailed analysis of competitor content related to the topic a user is interested in."},
        {"role": "user", "content": f"Considering the content analysis provided, develop a comprehensive content plan. The plan should include:\n\n"
                                f"Topic: {topic_query}\n"
                                f"An outline with a hierarchical structure of headings and subheadings that logically organize the content.\n\n"
                                f"Incorporate these SEO keywords: {final_keywords_df}. Ensure these keywords are naturally integrated into the headings and subheadings where relevant.\n\n"
                                f"While developing the plan, make sure to:\n"
                                f"Address the common topics and subtopics identified in the content analysis.\n"
                                f"Highlight any areas with contradicting viewpoints, and suggest a balanced approach to these topics.\n\n"
                                f"CONTENT ANALYSIS:\n {content_analysis}"
        }
    ]
    return interact_with_ai(messages_content_plan)

def create_content_draft(content_plan, content_analysis):
    messages_content_draft = [
        {"role": "system", "content": "You are a skilled content writer specializing in crafting engaging, informative, and SEO-friendly blog posts. You excel at following detailed content plans and adapting your writing style to meet specific guidelines and objectives. You have access to a content plan and an analysis of competitor content related to the topic a user is interested in."},
        {"role": "user", "content": f"Using the provided Content Plan and the insights from the Competitor Content Analysis, write a comprehensive article. Focus on delivering high-quality content that is engaging, informative, and optimized for search engines. Adhere to the structure and guidelines set in the Content Plan, and ensure the article addresses the topics and keywords specified. The article should be written in a style that is accessible and appealing to the target audience, while also being mindful of SEO best practices. Please provide the article only, without any additional commentary or explanations.\n\n"
                                f"Content Plan:\n {content_plan}\n\n"
                                f"Competitor Content Analysis:\n {content_analysis}"
        }
    ]
    return interact_with_ai(messages_content_draft)

def proofread_content_draft(content_draft, content_plan, content_analysis):
    messages_proofread_draft = [
        {"role": "system", "content": "You are an expert content editor with a keen eye for detail, specializing in refining and polishing written content. You excel at ensuring content is engaging, error-free, and adheres to SEO best practices. You have access to a draft article, its corresponding content plan, and an analysis of competitor content related to the topic a user is interested in."},
        {"role": "user", "content": f"Review the provided Content Draft, ensuring it aligns with the Content Plan and surpasses the quality of competitor content as detailed in the Competitor Content Analysis. Your task is to refine the draft, focusing on enhancing its engagement, clarity, and readability. Ensure the content is free of grammatical errors, follows SEO best practices, and is well-structured. Make any necessary adjustments to improve the overall quality and impact of the article. Please provide the revised article only, without any additional commentary or explanations.\n\n"
                                f"Content Draft:\n {content_draft}\n\n"
                                f"Content Plan:\n {content_plan}\n\n"
                                f"Competitor Content Analysis:\n {content_analysis}"
        }
    ]
    return interact_with_ai(messages_proofread_draft)

def provide_seo_recommendations(proofread_draft, final_keywords_df):
    messages_seo_recommendations = [
        {"role": "system", "content": "You are a seasoned SEO expert specializing in optimizing blog articles for search engines. You are adept at crafting compelling title tags and meta descriptions that improve click-through rates and accurately reflect the content. You have access to the final version of a blog article and a list of its targeting keywords related to the topic a user is interested in."},
        {"role": "user", "content": f"Examine the provided Content and the list of Targeting Keywords. Develop an optimized URL slug for the article. Generate three variations of a Title Tag, each designed to capture attention and encourage clicks. Additionally, create three variations of a Meta Description that accurately summarize the article's content and entice users to read further. Ensure each suggestion is SEO-friendly and aligns with current best practices. Please provide only the URL slug, Title Tags, and Meta Descriptions, without any additional commentary or explanations.\n\n"
                                f"Content:\n {proofread_draft}\n\n"
                                f"Targeting Keywords:\n {final_keywords_df}\n"
        }
    ]
    return interact_with_ai(messages_seo_recommendations)

def compile_final_deliverable(proofread_draft, seo_recommendations, final_keywords_df, df_serp, content_analysis):
    messages_final_deliverable = [
        {"role": "system", "content": "You are a meticulous Senior Project Manager with expertise in presenting comprehensive project deliverables. You excel at organizing and summarizing complex information into a clear, concise, and client-ready format. You have access to all the outputs generated during a content creation process related to the topic a user is interested in."},
        {"role": "user", "content": f"Compile the following information into a well-structured document for client presentation. The document should clearly outline the entire content generation process and include: \n\n"
                                f"- Title & Meta Description: Present the SEO-optimized title and meta description options, highlighting the chosen or recommended ones. Include alternative options for consideration.\n"
                                f"- URL: Provide the finalized URL slug for the article.\n"
                                f"- Targeting Keywords: List the primary keywords targeted in the content, along with their search volume.\n"
                                f"- Competitors: Summarize key information about the top competitors (Position, Link, and Title only), derived from the SERP analysis.\n"
                                f"- Notes: Offer insights into the content strategy, explaining what aspects are covered, unique points not addressed by competitors, and areas that may require human validation or review for accuracy and completeness.\n"
                                f"- Final Content: Present the fully proofread and polished article.\n\n"
                                f"Ensure the deliverable is client-friendly, easy to understand, and provides a comprehensive overview of the project. Please provide the final deliverable document only, without any additional commentary or explanations.\n\n"
                                f"Content:\n {proofread_draft}\n\n"
                                f"SEO Recommendations:\n {seo_recommendations}\n"
                                f"Targeting Keywords:\n {final_keywords_df}\n"
                                f"Competitors:\n {df_serp}\n"
                                f"Competitors Analysis:\n {content_analysis}\n"
        }
    ]
    return interact_with_ai(messages_final_deliverable)

app = Flask(__name__)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')  # Create index.html later

@app.route('/progress')
def progress():
    topic_query = request.args.get('topic')
    def generate():
        yield format_sse_message("start", message=f"Starting content generation for topic: {topic_query}")

        ## SERP Retrieval
        yield format_sse_message("progress", step="serp", message="Retrieving SERP Data...")
        df_serp = get_serpapi_data(topic_query)
        if df_serp.empty:
            error_message = "Failed to retrieve data from SerpAPI."
            yield format_sse_message("complete", step="serp", title="Could not retrieve SERP data.", data=error_message)
            return
        
        serp_data = []
        for index, row in df_serp.iterrows():
            serp_data.append({
                'Position': row['Position'],
                'Link': row['Link'],
                'Title': row['Title']
            })

        yield format_sse_message("complete", step="serp", title="SERP Data Retrieved", data=serp_data)


        ## SEMRush Retrieval
        yield format_sse_message("progress", step="semrush", message="Processing SEMRush Data...")
        df_results = df_serp.copy()

        semrush_data_list = process_semrush_data(df_results)

        semrush_data = []
        for index, row in df_results.iterrows():
            semrush_data.append({
                'Position': row['Position'],
                'Link': row['Link'],
                'Title': row['Title'],
                'SEMRush Data': row['SEMRush_Data']
            })

        common_keywords = []
        for index, row in semrush_data_list.iterrows():
            common_keywords.append({
                'Keyword': row['Keyword'],
                'Frequency': row['Frequency'],
                'Search Volume': row['Search Volume']
            })

        yield format_sse_message("complete", step="semrush", title="SEMRush Data Retrieved and Processed",
                               data={"semrush_results": semrush_data, "common_keywords": common_keywords})
    
        
        ## Fetching Content
        yield format_sse_message("progress", step="content", message="Starting content fetch...")
        
        content_fetch_results = []
        total_urls = len(df_results)

        for index, row in df_results.iterrows():
            url = row['Link']
            # Notify that we're starting to fetch this URL
            yield format_sse_message("progress", step="content", 
                                   message=f"Fetching content from {url}",
                                   current=index + 1,
                                   total=total_urls)
            # Force flush the message
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()
            
            content = fetch_content(url)
            success = not content.startswith("ERROR:")
            
            content_fetch_results.append({
                'Position': row['Position'],
                'Link': url,
                'Title': row['Title'],
                'SEMRush Data': row['SEMRush_Data'],
                'Content': content,
                'Success': success
            })
            
            # Send individual URL completion status
            yield format_sse_message("url_complete", 
                                   url=url,
                                   success=success,
                                   current=index + 1,
                                   total=total_urls)
            # Force flush the message
            if hasattr(sys.stdout, 'flush'):
                sys.stdout.flush()

        # Send final completion message
        yield format_sse_message("complete", step="content", 
                               title="Content Fetching Complete", 
                               data=content_fetch_results)


        yield format_sse_message("progress", step="analysis", message="Analyzing Content...")
        content_analysis = perform_content_analysis(content_fetch_results, topic_query)
        yield format_sse_message("complete", step="analysis", title="Analyzing Content", data=content_analysis)

        yield format_sse_message("progress", step="plan", message="Generating Content Plan...")
        content_plan = generate_content_plan(content_analysis, topic_query, semrush_data)
        yield format_sse_message("complete", step="plan", title="Content Planning", data=content_plan)

        yield format_sse_message("progress", step="draft", message="Creating Content Draft...")
        content_draft = create_content_draft(content_plan, content_analysis)
        yield format_sse_message("complete", step="draft", title="Content Draft", data=content_draft)

        yield format_sse_message("progress", step="proofread", message="Proofreading Content...")
        proofread_draft = proofread_content_draft(content_draft, content_plan, content_analysis)
        yield format_sse_message("complete", step="proofread", title="Proofreading", data=proofread_draft)

        yield format_sse_message("progress", step="seo", message="Generating SEO Recommendations...")
        seo_recommendations = provide_seo_recommendations(proofread_draft, semrush_data)
        yield format_sse_message("complete", step="seo", title="SEO Recommendations", data=seo_recommendations)

        yield format_sse_message("progress", step="final", message="Compiling Final Deliverable...")
        final_deliverable = compile_final_deliverable(proofread_draft, seo_recommendations, semrush_data, df_serp, content_analysis)
        yield format_sse_message("complete", step="final", title="Final Deliverable", data=final_deliverable)

    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=81, debug=True)
