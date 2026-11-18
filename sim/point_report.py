from pathlib import Path
import json,statistics as st
base=Path(__file__).resolve().parent;out=base.parent/'data/point-comparison';rows=json.loads((out/'结果.json').read_text())
s=(base/'turn_replay.template.html').read_text().replace('转向代价 · 同盘回放','八分区 · 同盘回放').replace('把转向算进路线，能多送几枚棋？','得分区分八段后，棋子送到了哪里？').replace('箭头左边只按路程规划，右边同时考虑转向。两边实际执行都要花时间转向。','左侧是上一版连续得分线，右侧是八个分区中心；两组都计算转向代价。').replace('原损失：仅路程','对照：连续得分线').replace('新损失：路程＋转向','本次：八分区中心').replace('每 10 秒得分按各局实际用时折算。','每 10 秒得分按固定 120 秒窗口折算。点模型没有车体或推板；抵达后停留 0.5 秒即交付，不模拟直退、碰撞或堆叠。').replace('每 50 cm','每 50 cm').replace('两侧时钟同步。','两侧时钟同步。右图数字为实际交付到各区的累计枚数，红色表示后续关闭；1 区在下，8 区在上。').replace('for(let [u,v] of row.points)',"""if(i===1){let counts=Array(8).fill(0);for(let [ts,j,n,v] of r.deliveries)if(ts<=t)counts[j]=v;for(let j=0;j<8;j++){let y=.42+j*.14375;line(.025,y+.003,.025,y+.14,counts[j]>=4?'#bb4734':'#24876b',8);ctx.fillStyle='#222';ctx.font='bold 14px sans-serif';ctx.fillText(counts[j],11,404-(y+.072)*200)}}for(let [u,v] of row.points)""").replace('<details>', '<div id="trips"></div><details>').replace("el('diff').textContent=", """el('trips').textContent='截至当前的交付记录：'+(rs[1].deliveries.filter(e=>e[0]<=t).map(([ts,j,n,total])=>ts.toFixed(1)+'s → '+(j+1)+'区 +'+n+'（共'+total+'）').join('；')||'尚无');el('diff').textContent=""")
summary=''
for m,name in zip(['full','passive','active'],['全图','被动','主动']):
    summary+='<tr><th>'+name+'</th>'
    for fn in [lambda r:r['score'],lambda r:sum(e[1]-e[0] for e in r['trace'] if e[6]=='turn'),lambda r:r['distance'],lambda r:r['score']/12]:
        a,b=[st.mean(fn(row['runs'][m][i]) for row in rows) for i in range(2)];summary+=f'<td>{a:.2f} → {b:.2f}</td>'
    summary+='</tr>'
    print(name,[(st.mean(row['runs'][m][i]['score'] for row in rows)) for i in range(2)],'overshoot',sum(max(row['runs'][m][1]['counts'])>4 for row in rows))
(out/'路线回放.html').write_text(s.replace('@@SUMMARY@@',summary).replace('@@DATA@@',json.dumps(rows,separators=(',',':'))))
