import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any


def load_val_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    读取验证数据集
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        包含验证数据的列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def load_test_cases() -> List[Dict[str, Any]]:
    """
    创建测试用例，包含不同类型的文案质量对比
    """
    test_cases = [
        {
            "name": "使用反馈类文案 - 科学专业 vs 夸张营销",
            "prompt": "请为\"原素方程\"品牌的如下产品创作一条侧重使用反馈的小红书文案\n\n{\"product_name\": \"净颜焕采精华\", \"category\": \"精华\", \"core_ingredients\": \"10%烟酰胺, 锌PCA, 红没药醇\", \"features\": \"高浓度烟酰胺, 源头控油, 改善暗沉\", \"target_audience\": \"油性痘痘肌\"}",
            "chosen": "油痘肌的夏日救星✨ 这支精华让我告别'反光脸'！\n\n每到夏天，我的T区就像个小型油田🌋 妆后2小时就开始脱妆，毛孔里还能挤出白色角栓...直到遇到原素方程的净颜焕采精华，终于明白什么叫源头控油的科学配方。\n\n🧪核心成分拆解：\n• 10%烟酰胺：不是盲目堆浓度！这个黄金配比既能抑制皮脂腺活跃度，又不会刺激屏障（实测比某些15%产品更温和）\n• 锌PCA：像吸油纸一样吸附表面油脂，但不会拔干\n• 红没药醇：给躁动的痘痘肌'灭火'，我经期爆痘时厚涂它，红肿消退速度肉眼可见\n\n💡使用Tips：\n洁面后第一步用，半滴管就能照顾全脸。凝露质地秒吸收，后续跟防晒也不会搓泥。用了三周，最惊喜的是下午脸颊不再暗沉得像隔夜菜，鼻翼的毛孔居然有'隐形'趋势！\n\n油皮姐妹注意⚠️ 如果正在用A醇类产品，建议错开早晚使用，避免叠加刺激～\n\n#科学护肤 #油皮精华实测 #屏障修护 #原素方程 #成分党",
            "rejected": "姐妹们！我发现了油痘肌的救星！！！原素方程'净颜焕采精华'真的绝绝子啊！！！😱😱😱\n\n昨晚睡前涂了它，今天醒来直接惊呆！！！脸上油田直接关闸！！！那个10%烟酰胺简直神了！！！三天！！！就三天！！！我的陈年痘印居然淡到几乎看不见！！！红没药醇直接把我泛红的脸蛋子安抚得像婴儿屁股一样嫩！！！\n\n什么控油啊、提亮啊，在它面前都是弟弟！！！现在我的脸就像开了磨皮特效！！！闺蜜追着我问是不是偷偷做了光子嫩肤！！！\n\n油痘肌的宝子们听我的！！！冲就完事了！！！这瓶精华就是你们的命运转折点！！！一夜回春不是梦啊！！！yyds我直接锁死！！！💥💥💥"
        },
        {
            "name": "科普知识类文案 - 专业科普 vs 简单描述",
            "prompt": "请为\"原素方程\"品牌的如下产品创作一条科普知识的小红书文案\n\n{\"product_name\": \"净颜焕采精华\", \"category\": \"精华\", \"core_ingredients\": \"10%烟酰胺, 锌PCA, 红没药醇\", \"features\": \"高浓度烟酰胺, 源头控油, 改善暗沉\", \"target_audience\": \"油性痘痘肌\"}",
            "chosen": "🔍 护肤小课堂：为什么你控油祛痘总是失败？\n\n油皮的朋友们，是不是总觉得脸越控越油，痘痘此起彼伏？今天我们来聊聊护肤界公认的'油痘肌救星'——烟酰胺。\n\n💡 烟酰胺是什么？\n它是维生素B3的衍生物，临床实证能：\n1️⃣ 抑制皮脂腺过度活跃（源头控油）\n2️⃣ 减少油脂氧化导致的暗沉（提亮肤色）\n3️⃣ 增强皮肤屏障功能（减少反复长痘）\n\n⚠️ 但要注意：\n• 浓度＜5%效果微弱\n• 需搭配锌类成分协同控油\n• 敏感肌建议从2%开始建立耐受\n\n✨ 科学实践案例：原素方程「净颜焕采精华」\n配方亮点：\n▫️10%黄金浓度烟酰胺（经皮吸收最佳区间）\n▫️锌PCA精准吸附多余油脂\n▫️红没药醇舒缓控油后的泛红\n\n没有酒精/香精的刺激配方，用精简的6种有效成分，实现控油+褪红+提亮三重目标。\n\n📚 知识点总结：\n控油不是一味去角质，而是通过调节皮脂腺功能（烟酰胺）+物理吸附（锌）+抗炎（红没药醇）的科学组合拳。\n\n#护肤科普 #油皮护肤 #成分党 #科学护肤 #原素方程",
            "rejected": "净颜焕采精华是一款针对油性痘痘肌设计的精华产品，其核心成分包括10%烟酰胺、锌PCA和红没药醇。其中，10%烟酰胺作为高浓度成分，能够有效控制皮脂分泌，从根本上减少痘痘的产生，同时改善肌肤暗沉，使肤色更加均匀明亮。锌PCA则有助于调节皮肤油脂平衡，红没药醇具有良好的舒缓和抗炎作用，能够缓解痘痘引发的红肿问题。这款产品专为追求清爽肤感和改善肌肤状态的油性肌肤用户设计。"
        },
        {
            "name": "破除误区类文案 - 理性分析 vs 模糊表述",
            "prompt": "请为\"原素方程\"品牌的如下产品创作一条破除护肤误区的小红书文案\n\n{\"product_name\": \"净颜焕采精华\", \"category\": \"精华\", \"core_ingredients\": \"10%烟酰胺, 锌PCA, 红没药醇\", \"features\": \"高浓度烟酰胺, 源头控油, 改善暗沉\", \"target_audience\": \"油性痘痘肌\"}",
            "chosen": "10%烟酰胺就是烂脸密码？高浓度≠高功效的残酷真相\n\n护肤圈对'高浓度'的盲目崇拜该停停了。\n\n看到10%烟酰胺就无脑冲？先回答三个问题：\n1. 你的屏障能承受这个浓度吗？\n2. 配方是否有缓冲体系？\n3. 其他成分是协同还是拖后腿？\n\n科学事实：\n◾️ 5%烟酰胺已能实现80%控油效果\n◾️ 超过8%浓度时刺激性呈指数级上升\n◾️ 锌PCA的控油协同作用比单纯堆浓度更重要\n\n原素方程净颜焕采精华的解题思路：\n✔️ 10%烟酰胺+锌PCA形成控油闭环\n✔️ 红没药醇即时中和刺激性\n✔️ 剔除所有着色剂/酒精等刺激源\n\n油痘肌的控油方案应该像精准化疗：在清除病灶（过量皮脂）的同时，最大限度保护健康组织（屏障）。那些让你'痛就对了'的产品，本质上都是配方不及格。\n\n#科学护肤 #油皮自救 #护肤真相大揭秘 你被高浓度产品坑过吗？",
            "rejected": "是不是高浓度就是高效？并非如此，高浓度的成分如果使用不当，反而可能加重皮肤负担。例如，过高的烟酰胺浓度会导致皮肤敏感，无法长期保持肌肤健康。而原素方程净颜焕采精华则采用了科学配比，其中10%的烟酰胺浓度既能有效控制油脂分泌、改善暗沉，又不会导致皮肤刺激，锌PCA和红没药醇的加入更是为肌肤提供了额外的舒缓与修复。"
        },
        {
            "name": "个人故事类文案 - 真实体验 vs 空洞描述",
            "prompt": "请为\"原素方程\"品牌的如下产品创作一条个人使用故事的小红书文案\n\n{\"product_name\": \"净颜焕采精华\", \"category\": \"精华\", \"core_ingredients\": \"10%烟酰胺, 锌PCA, 红没药醇\", \"features\": \"高浓度烟酰胺, 源头控油, 改善暗沉\", \"target_audience\": \"油性痘痘肌\"}",
            "chosen": "最近工作压力大，皮肤状态也跟着遭殃。作为一个油痘肌，额头和下巴的闭口此起彼伏，T区油光闪亮到能当镜子照。😭\n\n那天下班后和闺蜜约饭，她盯着我的脸突然说：\"你的皮肤怎么这么暗沉？\" 镜子里的自己确实像蒙了一层灰。回家后，我决定认真对待这个问题。\n\n在做了很多功课之后，我选择了原素方程的净颜焕采精华。吸引我的是它精简有效的配方：10%烟酰胺浓度刚刚好，锌PCA能源头控油，红没药醇温和修护。没有花哨的包装，白色瓶身上简单的标签透着科研感的严谨。\n\n连续使用两周后，最明显的变化是早上醒来时，脸上不再泛着油光。以前每到下午就必须用吸油纸，现在这种需求少了很多。更让我惊喜的是，那些顽固的闭口慢慢平复了，脸颊的毛孔看起来也干净了许多。\n\n现在每天护肤时，滴两滴精华在掌心，轻轻按压上脸的那一刻，像是在给皮肤做一个简单而有效的科学护理。它没有立竿见影的\"奇迹\"，但正是这种温和而持续的改善，让我重新找回了肌肤的平衡状态。\n\n有时候，解决问题不需要复杂的步骤，找到对的配方，给皮肤最需要的，就够了。\n\n#油痘肌护理 #精简护肤 #科学配方 #控油精华",
            "rejected": "我原来面部经常红肿，油脂分泌旺盛，肤色也不够明亮。自从用了净颜焕采精华后，这款产品含有10%烟酰胺，能够有效控制油脂分泌，同时还加入了锌PCA和红没药醇，帮助舒缓肌肤，减少红肿。现在我的肌肤状态好了很多，肤色也更加均匀透亮了。"
        },
        {
            "name": "电商介绍类文案 - 专业详细 vs 简单罗列",
            "prompt": "请为\"原素方程\"品牌的如下产品创作一条详细的电商平台产品介绍文案\n\n{\"product_name\": \"净颜焕采精华\", \"category\": \"精华\", \"core_ingredients\": \"10%烟酰胺, 锌PCA, 红没药醇\", \"features\": \"高浓度烟酰胺, 源头控油, 改善暗沉\", \"target_audience\": \"油性痘痘肌\"}",
            "chosen": "【原素方程 | 净颜焕采精华】科学配方，精准解决油痘肌困扰\n\n油性痘痘肌的困扰，源于皮脂过度分泌与屏障功能失衡。原素方程研发团队从科学角度出发，针对这一肌肤问题，设计了这款「净颜焕采精华」。我们摒弃复杂成分堆砌，以精简有效的配方，从源头改善肌肤状态。\n\n科学配方解析\n1. 10%浓度烟酰胺：经临床验证的有效浓度，能显著抑制皮脂腺活跃度，从源头减少油脂分泌。同时促进角质代谢，改善毛孔堵塞问题。\n2. 锌PCA：具有出色的控油与抑菌功效，协同烟酰胺调节皮脂膜健康状态。\n3. 红没药醇：温和舒缓成分，缓解因痘痘引发的肌肤不适，修护受损屏障。\n\n产品作用机理\n这款精华通过靶向调节皮脂腺功能，实现长效控油；同时优化肌肤微环境，减少痘痘反复发生。烟酰胺的加入还能阻断黑色素转运，逐步改善因出油氧化导致的肤色暗沉问题。\n\n使用建议\n建议每日早晚洁面后使用，配合轻柔按摩促进吸收。配方经过严格测试，温和不刺激，但首次使用高浓度烟酰胺产品建议先建立耐受。\n\n原素方程坚持用科学说话，每一款产品都经过反复验证。净颜焕采精华不承诺立竿见影的效果，但坚持使用28天，你将看到肌肤状态的明显改善。",
            "rejected": "这款产品是净颜焕采精华，属于精华类产品。其核心成分包括10%烟酰胺、锌PCA和红没药醇。这款精华通过高浓度烟酰胺有效控制油脂分泌，从源头改善肌肤暗沉问题，特别适合油性痘痘肌使用。"
        }
    ]
    return test_cases


def load_reward_model(model_path: str):
    """
    加载奖励模型和分词器
    """
    print(f"--- 正在加载奖励模型: {model_path} ---")
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        
        # 检查是否有可用的 GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()  # 设置为评估模式
        
        print(f"--- 模型加载成功，运行在: {device} ---")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None, None, None


def calculate_reward_score(model, tokenizer, device, prompt: str, response: str) -> float:
    """
    计算单个回答的奖励分数
    """
    # 构建对话消息
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    # 应用聊天模板并转换为tensor
    try:
        input_text = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=False, 
            return_tensors="pt"
        ).to(device)
    except Exception as e:
        # 如果聊天模板失败，使用简单格式
        input_text = tokenizer(
            f"用户: {prompt}\n助手: {response}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)
    
    # 模型推理
    with torch.no_grad():
        score = model(input_text).logits[0].item()
    
    return score


def run_single_test(model, tokenizer, device, test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行单个测试用例
    """
    print(f"\n{'='*60}")
    print(f"测试用例: {test_case['name']}")
    print(f"{'='*60}")
    
    # 计算分数
    score_chosen = calculate_reward_score(
        model, tokenizer, device, 
        test_case['prompt'], test_case['chosen']
    )
    
    score_rejected = calculate_reward_score(
        model, tokenizer, device, 
        test_case['prompt'], test_case['rejected']
    )
    
    # 判断结果
    is_correct = score_chosen > score_rejected
    score_diff = score_chosen - score_rejected
    
    print(f"好的回答分数: {score_chosen:.4f}")
    print(f"坏的回答分数: {score_rejected:.4f}")
    print(f"分数差异: {score_diff:.4f}")
    print(f"判断结果: {'✅ 正确' if is_correct else '❌ 错误'}")
    
    return {
        "test_name": test_case['name'],
        "score_chosen": score_chosen,
        "score_rejected": score_rejected,
        "score_diff": score_diff,
        "is_correct": is_correct
    }


def evaluate_model_on_dataset(model, tokenizer, device, dataset: List[Dict[str, Any]], model_name: str):
    """
    在验证集上评估模型
    
    Args:
        model: 奖励模型
        tokenizer: 分词器
        device: 设备
        dataset: 验证数据集
        model_name: 模型名称
        
    Returns:
        评估结果字典
    """
    print(f"\n--- 开始评估 {model_name} ---")
    
    chosen_scores = []
    rejected_scores = []
    
    for i, item in enumerate(dataset):
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # 计算chosen的分数
        chosen_score = calculate_reward_score(model, tokenizer, device, prompt, chosen)
        chosen_scores.append(chosen_score)
        
        # 计算rejected的分数
        rejected_score = calculate_reward_score(model, tokenizer, device, prompt, rejected)
        rejected_scores.append(rejected_score)
        
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(dataset)} 条数据")
    
    # 计算平均值
    avg_chosen_score = sum(chosen_scores) / len(chosen_scores)
    avg_rejected_score = sum(rejected_scores) / len(rejected_scores)
    score_diff = avg_chosen_score - avg_rejected_score
    
    print(f"{model_name} 评估完成:")
    print(f"  正样本平均分数: {avg_chosen_score:.4f}")
    print(f"  负样本平均分数: {avg_rejected_score:.4f}")
    print(f"  分数差异: {score_diff:.4f}")
    
    return {
        "model_name": model_name,
        "avg_chosen_score": avg_chosen_score,
        "avg_rejected_score": avg_rejected_score,
        "score_diff": score_diff,
        "chosen_scores": chosen_scores,
        "rejected_scores": rejected_scores
    }


def run_reward_model_tests():
    """
    运行完整的奖励模型测试
    """
    print("=== 奖励模型对比测试开始 ===")
    
    # 配置路径
    base_model_path = "models/Skywork-Reward-V2-Qwen3-0.6B"  # 基础模型路径
    finetuned_model_path = "models/Skywork-Reward-V2-Qwen3-0.6B-finetuned"  # 微调后模型路径
    val_dataset_path = "data/val_dataset_final.jsonl"  # 验证数据集路径
    
    # 加载验证数据集
    print("正在加载验证数据集...")
    val_dataset = load_val_dataset(val_dataset_path)
    print(f"成功加载验证数据集: {len(val_dataset)} 条")
    
    # 加载基础模型
    print("\n正在加载基础模型...")
    base_model, base_tokenizer, device = load_reward_model(base_model_path)
    if base_model is None:
        print("基础模型加载失败，测试终止")
        return
    
    # 加载微调后模型
    print("\n正在加载微调后模型...")
    finetuned_model, finetuned_tokenizer, device = load_reward_model(finetuned_model_path)
    if finetuned_model is None:
        print("微调后模型加载失败，测试终止")
        return
    
    # 在验证集上评估基础模型
    base_results = evaluate_model_on_dataset(base_model, base_tokenizer, device, val_dataset, "基础模型")
    
    # 在验证集上评估微调后模型
    finetuned_results = evaluate_model_on_dataset(finetuned_model, finetuned_tokenizer, device, val_dataset, "微调后模型")
    
    # 对比结果
    print(f"\n{'='*60}")
    print("模型对比结果")
    print(f"{'='*60}")
    print(f"数据集大小: {len(val_dataset)} 条")
    print()
    print("基础模型:")
    print(f"  正样本平均分数: {base_results['avg_chosen_score']:.4f}")
    print(f"  负样本平均分数: {base_results['avg_rejected_score']:.4f}")
    print(f"  分数差异: {base_results['score_diff']:.4f}")
    print()
    print("微调后模型:")
    print(f"  正样本平均分数: {finetuned_results['avg_chosen_score']:.4f}")
    print(f"  负样本平均分数: {finetuned_results['avg_rejected_score']:.4f}")
    print(f"  分数差异: {finetuned_results['score_diff']:.4f}")
    print()
    print("改进情况:")
    chosen_improvement = finetuned_results['avg_chosen_score'] - base_results['avg_chosen_score']
    rejected_improvement = finetuned_results['avg_rejected_score'] - base_results['avg_rejected_score']
    diff_improvement = finetuned_results['score_diff'] - base_results['score_diff']
    
    print(f"  正样本分数变化: {chosen_improvement:+.4f}")
    print(f"  负样本分数变化: {rejected_improvement:+.4f}")
    print(f"  分数差异变化: {diff_improvement:+.4f}")
    
    if diff_improvement > 0:
        print(f"  ✅ 微调后模型表现更好，分数差异增加了 {diff_improvement:.4f}")
    else:
        print(f"  ❌ 微调后模型表现较差，分数差异减少了 {abs(diff_improvement):.4f}")
    
    print("=== 奖励模型对比测试完成 ===")


if __name__ == "__main__":
    try:
        run_reward_model_tests()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
