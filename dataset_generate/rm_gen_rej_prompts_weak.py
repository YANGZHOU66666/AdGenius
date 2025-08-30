def create_weak_review_social_media_prompt(product_info_str: str):
    return f"""请为“原素方程”品牌的以下产品，生成一条产品分享。
    
{product_info_str}
    
# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "最近在用原素方程的玫瑰焕采精华..."}}
"""

def create_weak_educational_social_media_prompt(product_info_str: str):
    return f"""请你科普产品中一种成分的作用，并介绍“原素方程”的这款产品。

{product_info_str}

# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "玫瑰提取物是这款..."}}
"""

def create_weak_myth_busting_social_media_prompt(product_info_str: str):
    return f"""请你先打破一个常见的护肤误区，再引出“原素方程”的这款产品是怎么更科学的解决的

{product_info_str}

# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "是不是高浓度就是高效?..."}}
"""

def create_weak_storytelling_social_media_prompt(product_info_str: str):
    return f"""请你讲一个具体使用场景，再引入“原素方程”的这款产品

{product_info_str}

# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "我原来面部经常红肿..."}}
"""

def create_weak_ecommerce_prompt(product_info_str: str):
    return f"""请你介绍“原素方程”的这款产品

{product_info_str}

# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "这款产品..."}}
"""

def create_weak_paid_ad_prompt_cta(product_info_str: str):
    return f"""请你为“原素方程”的这款产品撰写一条文案，遵循【钩子-核心利益点-行动号召 (Hook-Benefit-CTA)】框架。

{product_info_str}

# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "来看看原素方程的这款产品..."}}
"""

def create_weak_paid_ad_prompt_pas(product_info_str: str):
    return f"""请你为“原素方程”的这款产品撰写一条文案，遵循【痛点-放大-解决 (Problem-Agitate-Solve)】框架。

{product_info_str}

# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "我...角质堆积让皮肤粗糙暗沉..."}}
"""

def create_weak_paid_ad_prompt_bab(product_info_str: str):
    return f"""请你为“原素方程”的这款产品撰写一条文案，遵循【之前-之后-桥梁 (Before-After-Bridge)】框架。

{product_info_str}

# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "我之前面部经常红肿..."}}
"""

def create_weak_paid_ad_prompt_fab(product_info_str: str):
    return f"""请你为“原素方程”的这款产品撰写一条文案，遵循【特点-优势-好处 (Feature-Advantage-Benefit)】框架。

{product_info_str}

# 输出格式
请严格以JSON格式返回，只包含一个键 "output"，值为生成的文案字符串。
示例: {{"output": "这款产品..."}}
"""