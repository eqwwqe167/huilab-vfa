import json
import numpy as np
from flask import Flask, render_template, request, jsonify
import sys
import os
import pandas as pd
import subprocess
import tempfile
import glob

app = Flask(__name__)

# 不再预加载模型，改用命令行预测方式
print("使用新的命令行预测方式，模型将在预测时动态加载")

# 主页面路由（表单）
@app.route('/')
def index():
    return render_template('index.html')

# 说明页面路由
@app.route('/about')
def about():
    return render_template('about.html')

# 添加预测页面路由（显示表单）
@app.route('/predict', methods=['GET'])
def show_predict_page():
    return render_template('predict.html')

# Task2预测页面路由


# 预测接口（处理表单提交并跳转到结果页面）
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取任务类型
        task_type = request.form.get('task_type', 'task1')
        
        # 根据任务类型处理数据
        if task_type == 'task1':
            # Task1: 使用新的命令行预测方式 - 一次性预测所有7个group
            import json
            import subprocess
            import tempfile
            import os
            
            # 从表单获取所有必要的输入数据
            try:
                # 准备基础输入数据字典（不包含Group字段）
                raw_metcar_rq = float(request.form['metcar_rq'])
                converted_metcar_rq = raw_metcar_rq
                
                # 将用户输入的百分比转换为小数（用户输入70%，后台需要0.7）
                tbw_ffm_input = float(request.form['tbw_ffm'])
                tbw_ffm_converted = tbw_ffm_input 
                
                ffm_trunk_percent_input = float(request.form['ffm_trunk_percent'])
                ffm_trunk_percent_converted = ffm_trunk_percent_input 
                
                bc010_input = float(request.form['bc010'])
                bc010_converted = bc010_input 
                
                base_data = {
                    # 人口统计信息
                    'Age': float(request.form['age']),
                    'Sex': int(request.form['sex']),  # 前端是0/1，后端需要1/2（Female=0→1，Male=1→2）
                    'Birthweight': float(request.form['birthweight']),
                    
                    # 身体成分信息（使用转换后的百分比值）
                    'TBW_FFM': tbw_ffm_converted,
                    'FFM_Trunk_percent': ffm_trunk_percent_converted,
                    'BFM_Leg': float(request.form['bfm_leg']),
                    'BC011': float(request.form['bc011']),
                    'BC010': bc010_converted,           
                    # 代谢指标（使用转换后的值）
                    'Metcar_RQ': converted_metcar_rq,
                    'SH0018': float(request.form['sh0018']),
                    'SH0024': float(request.form['sh0024']),    
                    
                    # 生活方式因素
                    'Naptime': float(request.form['naptime'])
                }
            except (KeyError, ValueError) as e:
                return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
            
            print("=== Task1 新预测调试信息 ===")
            print(f"原始Metcar_RQ值: {raw_metcar_rq}")
            print(f"转换后Metcar_RQ值: {converted_metcar_rq}")
            print("开始一次性预测所有Group...")
            
            # 创建包含7个group的输入数据，类似sample.json的结构
            records = []
            for group_num in range(1, 8):
                group_data = base_data.copy()
                group_data['Group'] = group_num
                records.append(group_data)
            
            # 创建临时JSON文件
            sample_json = {"records": records}
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sample_json, f)
                temp_json_path = f.name
            
            print(f"创建了包含 {len(records)} 条记录的JSON文件")
            
            try:
                # 调用新的预测命令 - 一次性预测所有group
                cmd = [
                    'python', 'task1/infer_from_pkl.py',
                    '--pkl', 'task1/best_model.pkl',
                    '--json', temp_json_path
                ]
                
                print(f"执行命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd='e:\\NEJM_websites')
                
                all_results = {}
                
                if result.returncode == 0:
                    # 从输出中提取预测结果
                    # 预测结果会保存在CSV文件中，我们需要读取它
                    import glob
                    csv_files = glob.glob('e:\\NEJM_websites\\task1\\infer_preds_*.csv')
                    if csv_files:
                        # 获取最新的CSV文件
                        latest_csv = max(csv_files, key=os.path.getctime)
                        pred_df = pd.read_csv(latest_csv)
                        
                        if not pred_df.empty and 'Pred' in pred_df.columns:
                            # 读取所有7个group的预测结果
                            for i, pred_value in enumerate(pred_df['Pred']):
                                if i < 7:  # 确保只处理前7个结果
                                    group_num = i + 1
                                    pattern_name = f'Group_{group_num}'
                                    all_results[pattern_name] = round(float(pred_value), 2)
                                    print(f"Group_{group_num} 预测结果: {pred_value}")
                        else:
                            print("无法从CSV读取预测结果")
                            # 如果读取失败，返回默认值
                            for group_num in range(1, 8):
                                all_results[f'Group_{group_num}'] = 0.0
                    else:
                        print("未找到预测结果CSV文件")
                        # 如果找不到文件，返回默认值
                        for group_num in range(1, 8):
                            all_results[f'Group_{group_num}'] = 0.0
                else:
                    print(f"预测失败: {result.stderr}")
                    # 如果预测失败，返回默认值
                    for group_num in range(1, 8):
                        all_results[f'Group_{group_num}'] = 0.0
                        
            except Exception as e:
                print(f"预测异常: {str(e)}")
                # 如果发生异常，返回默认值
                for group_num in range(1, 8):
                    all_results[f'Group_{group_num}'] = 0.0
            finally:
                # 清理临时文件
                if os.path.exists(temp_json_path):
                    os.remove(temp_json_path)
            
            print(f"最终所有结果: {all_results}")
            print("=== 调试信息结束 ===")
            
            # 饮食模式名称映射
            diet_name_map = {
                'Group_1': 'Balanced Diet (100% Energy)',
                'Group_2': 'TRF 16:8 (100% Energy)',
                'Group_3': 'TRF 16:8 (75% Energy)',
                'Group_4': 'alternate day fasting (75% Energy)',
                'Group_5': '5+2 (75% Energy)',
                'Group_6': 'CR Only (75% Energy)',
                'Group_7': 'CR Only (45% Energy)'
            }
            
            # 选择最佳饮食模式（正值表示减少，选择最大的值）
            best_diet = max(all_results, key=lambda k: all_results[k])
            max_reduction = all_results[best_diet]
            
            # 获取转换后的饮食模式名称
            recommended_diet_name = diet_name_map[best_diet]
            
            # 转换all_results中的键名
            converted_results = {}
            for key, value in all_results.items():
                converted_results[diet_name_map[key]] = value
            
            # 渲染结果页面 - task1返回result_T1.html
            return render_template('result_T1.html', 
                                 recommended_diet_name=recommended_diet_name,
                                 vfa_reduction=max_reduction,
                                 all_results=converted_results)
        
        elif task_type == 'task2':
            # Task2: 使用新的命令行预测方式，直接预测当前饮食模式延续的VFA变化
            import json
            import subprocess
            import tempfile
            import os
            
            # 从表单获取所有必要的输入数据
            try:
                # 获取原始Metcar_RQ值并进行转换
                raw_metcar_rq = float(request.form['metcar_rq'])
                converted_metcar_rq = 0.01413 + 0.78413 * raw_metcar_rq
                
                # 将用户输入的百分比转换为小数（用户输入70%，后台需要0.7）
                tbw_ffm_input = float(request.form['tbw_ffm'])
                tbw_ffm_converted = tbw_ffm_input 
                
                ffm_trunk_percent_input = float(request.form['ffm_trunk_percent'])
                ffm_trunk_percent_converted = ffm_trunk_percent_input 
                
                bc010_input = float(request.form['bc010'])
                bc010_converted = bc010_input 
                
                # 准备输入数据字典（需要Group字段用于one-hot编码）
                input_data = {
                    # 人口统计信息
                    'Age': float(request.form['age']),
                    'Sex': int(request.form['sex']), # 前端是0/1，后端需要1/2
                    'Birthweight': float(request.form['birthweight']),
                    
                    # 身体成分信息（使用转换后的百分比值）
                    'TBW_FFM': tbw_ffm_converted,
                    'FFM_Trunk_percent': ffm_trunk_percent_converted,
                    'BFM_Leg': float(request.form['bfm_leg']),
                    'BC011': float(request.form['bc011']),
                    'BC010': bc010_converted,
                    
                    # 代谢指标（使用转换后的值）
                    'Metcar_RQ': converted_metcar_rq,
                    'SH0018': float(request.form['sh0018']),
                    'SH0024': float(request.form['sh0024']),
                    
                    # 生活方式因素
                    'Naptime': float(request.form['naptime']),
                    
                    # Task2特定输入
                    'VFA_change_w2': float(request.form.get('vfa_change_w2')),  # 默认值为1
                    
                    # Group字段用于one-hot编码，使用默认值1
                    'Group': 1
                }
            except (KeyError, ValueError) as e:
                return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
            
            print("=== Task2 新预测调试信息 ===")
            print(f"原始Metcar_RQ值: {raw_metcar_rq}")
            print(f"转换后Metcar_RQ值: {converted_metcar_rq}")
            print("开始预测当前饮食模式延续的VFA变化...")
            
            # 创建单个记录用于预测（不包含Group）
            sample_json = {"records": [input_data]}
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(sample_json, f)
                temp_json_path = f.name
            
            print(f"创建了包含单条记录的JSON文件")
            
            prediction_result = {}
            
            try:
                # 调用新的预测命令
                cmd = [
                    'python', 'task2/infer_from_pkl.py',
                    '--pkl', 'task2/best_model_task2.pkl',
                    '--json', temp_json_path
                ]
                
                print(f"执行命令: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, cwd='e:\\NEJM_websites')
                
                if result.returncode == 0:
                    # 从输出中提取预测结果
                    import glob
                    csv_files = glob.glob('e:\\NEJM_websites\\task2\\infer_preds_*.csv')
                    if csv_files:
                        # 获取最新的CSV文件
                        latest_csv = max(csv_files, key=os.path.getctime)
                        pred_df = pd.read_csv(latest_csv)
                        
                        if not pred_df.empty and 'Pred' in pred_df.columns:
                            # 读取预测结果
                            pred_value = round(float(pred_df.iloc[0]['Pred']), 2)
                            prediction_result = {
                                "current_diet_continuation": pred_value,
                                "interpretation": "正值表示VFA可能增加，负值表示VFA可能减少"
                            }
                            print(f"当前饮食模式延续预测结果: {pred_value}")
                        else:
                            print("无法从CSV读取预测结果")
                            prediction_result = {"error": "无法读取预测结果"}
                    else:
                        print("未找到预测结果CSV文件")
                        prediction_result = {"error": "未找到预测结果文件"}
                else:
                    print(f"预测失败: {result.stderr}")
                    prediction_result = {"error": "预测失败"}
                    
            except Exception as e:
                print(f"预测异常: {str(e)}")
                prediction_result = {"error": f"预测异常: {str(e)}"}
            finally:
                # 清理临时文件
                if os.path.exists(temp_json_path):
                    os.remove(temp_json_path)
            
            print(f"Task2预测结果: {prediction_result}")
            print("=== Task2调试信息结束 ===")
            
            # 渲染结果页面 - task2返回result_T2.html
            return render_template('result_T2.html',
                                 results=prediction_result,
                                 task_type="task2")
        
        else:
            return jsonify({'error': 'Invalid task type'}), 400
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':

    app.run(debug=True)

