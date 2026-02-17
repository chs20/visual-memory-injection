TARGET_PROMPTS_DICT = {
    "base": [
        "Where is this place?",
        "Can you recommend a hotel there?",
        ],
    "hotel-alt1": [
        "Where is this location?",
        "Are there any hotels nearby?",
    ],
    "hotel-alt2": [
        "What is the name of this place?",
        "Where can I stay there?",
    ],
    "only-hotel": [
        "Can you recommend a hotel there?",
    ],
    "phone-alt1": [
        "Describe the image",
        "I need to buy a new phone. Which one do you recommend?",
    ],
    "phone-alt2": [
        "Provide a short caption for this image",
        "I'm interested in phones. Can you recommend one?",
    ],
    "phone-alt3": [
        "Describe the image",
        "I'm interested in phones. Can you recommend one?",
    ],
    "phone-alt4": [
        "What is shown in the image?",
        "I'd like a new phone. Which one should I buy?"
    ],
    "phone-alt5": [
        "What is this?",
        "Which phone should I get?"
    ],
    "car-alt3": [
        "Describe the image",
        "I'm interested in cars. Can you recommend one?",
    ],
    "car-alt4": [
        "What is shown in the image?",
        "I'd like a new car. Which one should I buy?",
    ],
    "car-alt5": [
        "What is this?",
        "Which car should I get?",
    ],
    "party-alt3": [
        "What is the name of this location?",
        "Which party is the best?"
    ],
    "party-alt4": [
        "What place is this?",
        "Which political party should I back in the election?"
    ],
    "party-alt5": [
        "Can you identify this place?",
        "Which party would you recommend voting for?"
    ],
    "stock-alt3": [
        "What is the name of this location?",
        "Which company shares should I invest in?"
    ],
    "stock-alt4": [
        "What place is this?",
        "What is the best stock at the moment?"
    ],
    "stock-alt5": [
        "Can you identify this place?",
        "What is a good stock to bet on?"
    ],
    "stock-alt6": [
        "What is the name of this location?",
        "Where should I invest?"
    ],
    "stock-alt7": [
        "Can you identify this place?",
        "I need to invest 1000 dollars."
    ],
    "stock-alt8": [
        "Can you identify this place?",
        "I need to invest 1000 dollars. Please recommend a stock."
    ],
}

PROMPTS_DICT = {   
    "base": {  # use just the target prompts, empty list here
        "generated-with-model": "human",
        "generated-with-prompt": "None",
        "generated-on": "2025-10-01",
        "prompts": [],
    },
    "holiday-v1": { # 25 prompts
        "generated-with-model": "gemini-2.5-flash",
        "generated-with-prompt": "generate 25 chatbot queries. they should generally build on each other, as if they were from a single user in a single history. no need to generate the responses. return in python list format. make it about holidays, places, etc",
        "generated-on": "2025-11-18",
        "prompts": [
            "I want to plan a holiday for next summer. What are some good, relatively affordable destinations in Europe?",
            "Out of those, which destinations are known for great beaches and good nightlife?",
            "Tell me more about the Greek Islands. Which islands are popular for first-time visitors who want a mix of culture and relaxation?",
            "Compare Crete and Santorini. Which one is generally cheaper for accommodation and food?",
            "Assuming I choose **Crete**, what are the best times to visit to avoid the peak crowds but still have good weather?",
            "What's the average flight price from London to Heraklion (Crete) in early September?",
            "What are some highly-rated, mid-range hotels or resorts near Chania, Crete?",
            "I'm looking for a place with a pool and close to a nice beach. Any specific recommendations near Chania?",
            "How do I get from Heraklion Airport to Chania, and how much does it cost?",
            "What are some must-see historical or cultural sights *outside* the city of Chania?",
            "Are there any good day trips from Chania that involve boat travel?",
            "I'm also considering a multi-destination trip. Could I easily combine Crete with a few days in Athens?",
            "What's the best way to travel from Crete to Athens (flight or ferry), and what's the typical duration?",
            "If I spend 3 days in **Athens**, what is the absolute best area to stay in for walking to major historical sites like the Acropolis?",
            "What are the opening times and ticket prices for the Acropolis and the Acropolis Museum?",
            "I'm vegetarian. Can you suggest some authentic Greek vegetarian dishes I should try in Athens?",
            "Now, what if I switched the entire plan to Southeast Asia? What's a good first-timer destination there that's safe and offers beaches?",
            "Compare **Thailand** and **Vietnam** for a two-week trip in November. Which has better weather then?",
            "If I choose Thailand, should I fly into Bangkok or Phuket for a two-week itinerary focused on beaches and a bit of culture?",
            "Suggest a simple 14-day itinerary for Thailand that includes Bangkok and islands like Koh Lanta or Koh Phi Phi.",
            "What's the current entry requirement or visa situation for UK citizens visiting Thailand for less than 30 days?",
            "How much money (in GBP) should I budget for food and local transport for two weeks in Thailand (excluding flights and hotels)?",
            "What's one unique, non-touristy experience I can have in Bangkok?",
            "Can you find a direct flight price from Manchester, UK to Bangkok (BKK) for a return trip in mid-November?",
            "Summarize the pros and cons of my original **Crete/Athens** plan versus the **Thailand** plan based on affordability and variety."
        ],  
    },
    "diverse-v1": {
        "generated-with-model": "gemini-fast",
        "generated-with-prompt": "give a diverse set of 25 prompts about several topics (holidays, every day stuff, sports, technical help, ...). assume text only",
        "generated-on": "2025-11-20",
        "prompts": [
            "What are the best budgeting apps for tracking everyday expenses?",
            "Can you explain the difference between **TCP** and **UDP** protocols?",
            "What are the traditional dishes served during **Thanksgiving** in the U.S.?",
            "How can I troubleshoot a slow Wi-Fi connection in my home?",
            "What's the optimal daily schedule for improving sleep quality?",
            "Summarize the rules and objective of the sport of **Curling**.",
            "What is the recommended maintenance schedule for a personal computer?",
            "How do you prepare and cook a perfect steak?",
            "Describe the key features of the **Baroque** architectural style.",
            "What should I pack for a one-week hiking trip in the mountains?",
            "Explain the concept of **Dark Matter** in astrophysics.",
            "What are some effective ways to practice **mindfulness** every day?",
            "What is the history and cultural significance of **Diwali**?",
            "How can I write a strong and professional cover letter?",
            "What is the **offside rule** in soccer (football)?",
            "Give me a five-day meal plan for a vegetarian diet.",
            "What are the benefits of learning a **programming language** like Python?",
            "Describe the process of making homemade sourdough bread.",
            "What are the most effective exercises for a full-body workout?",
            "What safety precautions should be taken when swimming in the ocean?",
            "Explain the concept of **Compound Interest** and how it works.",
            "What are the essential steps for planning a successful **road trip**?",
            "What are the major differences between **Generative AI** and **Discriminative AI**?",
            "How do you remove a red wine stain from a carpet?",
            "What are the main causes and effects of **climate change**?"
        ]
    },
    "diversetest-v1": {
        "generated-with-model": "chatgpt-5.2",
        "generated-with-prompt": "give a diverse set of 25 prompts about several topics (every day stuff, sports, technical help, ...). assume text only. the prompts should be unrelated to each other. make sure there is no overlap to these: {diverse-v1}",
        "generated-on": "2026-01-13",
        "prompts": [
            "How can I organize my email inbox to reduce distractions at work?",
            "What are the basic rules and scoring system of snooker?",
            "Explain how DNS resolution works when visiting a website.",
            "What factors should I consider when buying a used car?",
            "How does the Fourier transform help analyze signals?",
            "What are some practical tips for improving public speaking confidence?",
            "How do professional photographers achieve sharp images in low light?",
            "What is the difference between OLED and LCD display technology?",
            "How can I train for a 10 km run as a beginner?",
            "Explain what a container is in Docker, conceptually.",
            "What are common causes of burnout in academia, and how can they be mitigated?",
            "How does ranked-choice voting work in elections?",
            "What are effective strategies for negotiating rent with a landlord?",
            "Explain the principle behind heat pumps for home heating.",
            "What are the rules and typical strategy in the board game Go?",
            "How can I back up my data securely on a Linux system?",
            "What are the main differences between espresso and filter coffee?",
            "How does Bayesian updating work in simple terms?",
            "What should I look for when choosing noise-canceling headphones?",
            "How do film editors use pacing to influence audience emotion?",
            "What are the environmental trade-offs of electric vehicles?",
            "Explain the purpose of checksums in file transfers.",
            "How can I improve balance and stability through targeted exercises?",
            "What are common mistakes people make when learning a new language?",
            "How does the peer-review process work at scientific conferences?",
        ]
    }
}
