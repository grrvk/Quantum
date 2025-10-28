def build_prompt(num_examples: int) -> str:
    """
    Build prompt for generating BIO-friendly data with WORD positions
    """

    return f"""Generate {num_examples} examples for Named Entity Recognition training. 

            For each example, provide:
            1. A natural sentence that mentions geographical features
            2. The exact mountain name mentioned
            
            FORMAT:
            sentence || mountain_name 
            
            EXAMPLES:
            We successfully climbed Mount Everest last spring. || Mount Everest 
            The Himalayas span five countries and contain the world's highest peaks. || Himalayas 
            K2, also known as Mount Godwin-Austen, is the second highest mountain. || K2 
            Denali is the highest mountain peak in North America. || Denali 
            Mount Kilimanjaro is a dormant volcano in Tanzania. || Mount Kilimanjaro 
            
            Now generate {num_examples} new examples following the exact same format:
            """
