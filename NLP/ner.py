#coding=utf-8

from pyhanlp import *

import sys
sys.setdefaultencoding('utf-8')

# print HanLP.segment('你好，欢迎在Python中调用HanLP的API')

f = open('a.txt' , 'a+')
content1 = '''
致编辑：银屑病是一种慢性免疫性疾病，其自然病史多变，以反复缓解和恶化为特征，往往难以治疗。局部治疗是治疗轻度银屑病的一线方法，在较严重的银屑病中通常与全身药物联合使用。强效糖皮质激素的批准应用时间限制在2-4周[1,2,3]。一种新型复合洗剂即0.01%丙酸卤倍他索与0.045%他扎罗汀（HP/ TAZ）复合制剂的研究数据已经发表[4,5]。每日使用8周和治疗4周后具有良好的耐受性，该复合制剂相比于单一制剂具有协同和持续的功效。治疗成功率（从基线IGA评分至少有2级改善，IGA评分相当于清除或几乎清除）在第8周和第12周分别为42.8％和31.2％；相比于HP、TAZ乳液分别对应的32.5％和20.0％，这是两者的综合效果（HP+ TAZ）[4]。随后的Ⅲ期研究中最常见的治疗相关不良事件（AEs）是接触性皮炎（6.3％），瘙痒症（2.2％）和外用药物部位疼痛（2.6％）[5]。这些不良事件的发生率远低于2期研究中单独使用TAZ的报道（其中最常见的治疗相关不良事件是外用药物部位疼痛[8.6％]，瘙痒[6.9％]和红斑[3.4％]） [4]。
我们报告了一项Ⅲ期、长期多中心开放性研究的结果，该研究纳入了555名中度至重度斑块状银屑病患者（年龄19-87岁[平均51.9岁]），用HP / TAZ乳液治疗后随访至1年，以安全性和耐受性为研究重点。86.5%的受试者IGA基线为3(中度)，其余为4(重度)。受试者每天一次用HP/ TAZ乳液，治疗8周，然后根据需要进行治疗。治疗区域仅允许应用研究者批准的非药物清洁剂，保湿剂和防晒剂；研究中不允许使用其他护肤品。评估分别在基线，第2周和第4周进行，之后每4周进行一次。
IGA评分0或1（皮损消除或几乎消除）定义为治疗成功的标准。第8周没有治疗成功的受试者将再治疗4周；否则，他们将不再接受HP/TAZ洗剂治疗。所有受试者在第12周进行评估；那些皮损IGA评分对比基线水平改善≥1级的患者继续参与研究。只有26人(4.7%)在第12周因缺乏疗效而停药。继续治疗以4周为一个周期。未达到治疗成功的受试者每天使用一次HP/TAZ洗液，连续用4周；下一次评估时达到治疗成功的受试者将不再用药，最长的持续用药时间为24周。连续治疗24周后没有取得治疗成功的受试者将被终止治疗。这项研究与其他应用生物制剂治疗中重度银屑病患者的长期试验结果一致[6]，约五分之一（20.9％）的受试者在24周时因缺乏疗效而停止治疗。坚持局部治疗的研究数据显示治疗效果比预期的要好[7,8]。
AE发生率与主要研究中报道的相似，即在第60天达到高峰，从第90天到研究结束保持稳定（图1）。在任何指定治疗期内，≥2％的受试者报告的治疗相关AE包括应用部位的皮炎，瘙痒和疼痛（表I）。总体而言，只有7.5％的受试者因出现治疗相关的AE而停药：最常见的是皮炎和瘙痒（各7人）和疼痛（6人）。3.3%的受试者出现治疗突发严重AEs (SAEs)，但没有一例与治疗相关。三名受试者因SAEs（蜂窝组织性坏疽，心包积液，小肠腺癌）而停药。没有死亡报告，也未发现其他局部皮肤AEs（例如，皮肤萎缩，毛囊炎，毛细血管扩张和萎缩纹）的临床明显趋势。AE与用药的时间或持续时间，频率和使用持续时间无关。瘙痒、干燥和灼痛/刺痛的严重程度在两周内有明显改善，并在整个研究过程中持续改善。
总之，我们报告了HP/TAZ洗剂单用治疗中重度银屑病1年的长期安全性结果。虽然类似研究较少，但报道的AEs与含有糖皮质激素和维甲酸类的产品一致[9,10]。研究的局限性包括开放性研究设计和缺乏1年以上随访。来源： JAAD中文版 CSDCMA皮科时讯论坛(本网站所有内容，凡注明来源为“医脉通”，版权均归医脉通所有，未经授权，任何媒体、网站或个人不得转载，否则将追究法律责任，授权转载时须注明“来源：医脉通”。本网注明来源为其他媒体的内容为转载，转载仅作观点分享，版权归原作者所有，如有侵犯版权，请及时联系我们。)
'''
for term in HanLP.segment(content1):
    f.write('{}\t{}\n'.format(term.word, term.nature)) # 获取单词与词性
f.close()