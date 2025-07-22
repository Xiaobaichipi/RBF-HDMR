import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
import geatpy as ea
from sklearn.preprocessing import StandardScaler
import sys
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_problem
from pymoo.factory import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.core.problem import ElementwiseProblem

# 打开日志文件
log_file = open('output_log.txt', 'w')

# 创建双重输出流
original_stdout = sys.stdout

class Logger:
    def write(self, text):
        original_stdout.write(text)  # 输出到控制台
        log_file.write(text)          # 写入文件
        log_file.flush()              # 确保立即写入
    def flush(self):
        original_stdout.flush()

class HDMR_RBF:
    """
    A class to implement the RBF-HDMR method for surrogate modeling.
    """

    def __init__(self, mode, path, yijie_file, erjie_file, save_model, test_file):
        """
        Initializes the HDMR_RBF object.

        Args:
            mode (str): Specifies whether to perform first-order ('first') or second-order ('second') modeling.
            path (str): The directory where data files are located.
            yijie_file (str): Filename for the first-order data (CSV).
            erjie_file (str): Filename for the second-order data (CSV).
            save_model (str): Path to save the trained first-order RBF-HDMR model parameters.
        """
        self.mode = mode
        self.path = path
        self.yijie_file = yijie_file
        self.erjie_file = erjie_file
        self.test_file = test_file
        self.save_model_path = save_model
        self.yijie_data = None
        self.erjie_data = None
        self.f0 = None  # 基准样本的目标函数值
        self.RBF_models = {}  # 存储一阶 RBF 模型
        self.linear_threshold = 1e-2  # 线性判断阈值
        self.coupling_threshold = 1e-2  # 耦合项判断阈值
        self.variables = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
        self.target_functions = ['ex1', 'ex3', 'ex4', 'ex5', 'ex6-fz', 'head']
        self.first_order_models = {}
        self.second_order_models = {}

    def load_first_order_data(self):
        """
        Loads first-order data from the specified CSV file.
        """
        try:
            file_path = os.path.join(self.path, self.yijie_file)
            self.yijie_data = pd.read_csv(file_path)
            print(f"一阶数据加载成功，路径为 {file_path}.")
        except FileNotFoundError:
            print(f"一阶数据加载失败，没有发现 {file_path}.")
            exit()
        except Exception as e:
            print(f"E无法加载一节数据: {e}")
            exit()

    def load_second_order_data(self):
        """
        Loads second-order data from the specified CSV file.
        """
        try:
            file_path = os.path.join(self.path, self.erjie_file)
            self.erjie_data = pd.read_csv(file_path)
            print(f"二阶数据加载成功： {file_path}.")
        except FileNotFoundError:
            print(f"没有发现二阶数据 {file_path}.")
            exit()
        except Exception as e:
            print(f"二阶数据加载失败: {e}")
            exit()

    def calculate_f0(self, target_function):
        """
        Calculates the baseline response value (f0) for a given target function.

        Args:
            target_function (str): The name of the target function column.
        """
        if self.yijie_data is None:
            print("错误：一阶数据没有加载，应当先调用load_first_order_data().")
            exit()
        try:
            self.f0 = self.yijie_data[self.yijie_data['标号'] == 0][target_function].values[0]
            print(f"基准响应值 (f0) calculated as {self.f0} for {target_function}.")
        except KeyError:
            print(f"Error: 目标函数 '{target_function}' 并不在一阶数据中.")
            exit()
        except Exception as e:
            print(f"无法计算 f0: {e}")
            exit()

    def construct_initial_sample_set(self, variable, target_function):
        """
        Constructs the initial sample set S_i^0 and calculates sample point response values.

        Args:
            variable (str): The name of the variable (e.g., 'V1').
            target_function (str): The name of the target function column.

        Returns:
            tuple: A tuple containing the initial sample set (X) and the corresponding response values (y).
        """
        if self.yijie_data is None:
            print("错误：一阶数据没有加载，应当先调用load_first_order_data().")
            exit()
        try:
            # 基准样本
            x0 = self.yijie_data[self.yijie_data['标号'] == 0][variable].values[0]

            # 变化变量的样本
            variable_data = self.yijie_data[self.yijie_data['变化变量'] == variable]
            xi_values = variable_data[variable].values

            # 获取目标函数值
            f_values = variable_data[target_function].values

            # 计算 f_i^L(x_i) 和 f_i^R(x_i)
            f_i_L = f_values[0] - self.f0  # 第一个样本
            f_i_R = f_values[-1] - self.f0  # 最后一个样本

            # 构建初始样本点集
            X = np.array(xi_values).reshape(-1, 1)  # 将xi_values转换为二维数组
            y = f_values - self.f0  # 计算样本点响应值

            # print(f"Initial sample set constructed for variable {variable} and target function {target_function}.")
            return X, y, x0, f_i_L, f_i_R
        except KeyError:
            print(
                f"Error: 变量 '{variable}' 或目标函数 '{target_function}' 没有在一阶数据中发现.")
            exit()
        except Exception as e:
            print(f"无法构建初始样本: {e}")
            exit()

    def build_rbf_model(self, X, y):
        """
        Builds a RBF model using scikit-learn's GaussianProcessRegressor with a Gaussian (RBF) kernel.

        Args:
            X (numpy.ndarray): The input sample set.
            y (numpy.ndarray): The corresponding response values.

        Returns:
            sklearn.gaussian_process.GaussianProcessRegressor: A trained GaussianProcessRegressor model, or None if an error occurs.
        """
        try:
            # 数据校验
            if X is None or y is None:
                print("Error: 输入数据X或y不存在.")
                return None

            if np.isnan(X).any() or np.isnan(y).any():
                print("Error: x或y中包含极大值.")
                return None

            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
                print("Error: X或y不是numpy数组.")
                return None

            if len(X) <= 1:
                print("Warning: 输入数据只有一个样本点，RBF无法拟合.")
                return None

            # 定义高斯过程回归模型，使用 RBF 核函数
            kernel = C(1.0, (1e-1, 1e-1)) * RBF(1.0, (1e-1, 1e1))  # 调整参数范围

            # 创建高斯过程回归模型对象
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)  # 增加优化器重启次数

            # 训练模型
            gp.fit(X, y)

            print("RBF模型构建成功.")
            return gp

        except Exception as e:
            print(f"RBF构建失败: {e}")
            return None

    def is_linear(self, variable, x0, f_i_L, f_i_R):
        """
        Determines whether the function f_i(x_i) is linear based on the given condition.

        Args:
            variable (str): The name of the variable.
            x0 (float): The baseline value of the variable.
            f_i_L (float): The response value at the left endpoint.
            f_i_R (float): The response value at the right endpoint.

        Returns:
            bool: True if the function is linear, False otherwise.
        """
        try:
            # 获取变化变量的样本
            variable_data = self.yijie_data[self.yijie_data['变化变量'] == variable]
            xi_values = variable_data[variable].values

            # 计算左端点和右端点
            x_L = xi_values[0]  # 左端点
            x_R = xi_values[-1]  # 右端点

            # 计算线性判断条件
            condition = abs((f_i_L / (x_L - x0)) - (f_i_R / (x_R - x0)))

            # 判断是否满足线性条件
            is_linear = condition < self.linear_threshold

            print(f"变量{variable}进行线性判断: Condition = {condition}, Linear = {is_linear}")
            return is_linear
        except Exception as e:
            print(f"无法检查变量{variable}的线性: {e}")
            return False

    def build_first_order_models(self, target_function):
        """
        Builds RBF-HDMR models for all non-coupled terms f_i(x_i).

        Args:
            target_function (str): The name of the target function column.
        """
        self.first_order_models = {}  # 清空之前的模型

        self.calculate_f0(target_function)  # 计算 f0
        # 保存 f0
        self.save_f0(target_function, self.f0)

        for variable in self.variables:
            # 针对V10，yijie.csv中样本只有6个，进行判断
            if variable == 'V10':
                variable_data = self.yijie_data[self.yijie_data['变化变量'] == variable]
                if len(variable_data) < 6:
                    # print(f"Skipping variable {variable} because it has less than 6 samples.")
                    continue

            X, y, x0, f_i_L, f_i_R = self.construct_initial_sample_set(variable, target_function)  # 构建初始样本集
            rbf_model = self.build_rbf_model(X, y)  # 构建 RBF 模型
            if rbf_model is None:
                is_linear = False  # 模型构建失败，设置为非线性
            else:
                is_linear = self.is_linear(variable, x0, f_i_L, f_i_R)  # 判断线性

            self.first_order_models[variable] = {
                'model': rbf_model,
                'is_linear': is_linear,
                'f0': self.f0
            }
            print(f"对变量{variable}构建一阶模型.")

    def validate_first_order_model(self, target_function):
        """
        Validates the first-order RBF-HDMR model.

        Args:
            target_function (str): The name of the target function column.

        Returns:
            bool: True if the model is valid, False otherwise.
        """
        if self.yijie_data is None:
            # print("Error: First-order data not loaded. Call load_first_order_data() first.")
            return False

        # 取每个变化变量的最后一条数据作为验证数据
        validation_data = {}
        for variable in self.variables:
            variable_data = self.yijie_data[self.yijie_data['变化变量'] == variable]
            if not variable_data.empty:
                validation_data[variable] = variable_data.iloc[-1][variable]
            else:
                validation_data[variable] = None  # 如果没有找到该变量的数据，则设置为 None

        # 计算实际函数响应值
        f_xe = self.yijie_data[self.yijie_data['标号'] == 0][target_function].values[0]  # 基准样本的目标函数值，这里假设基准样本为实际值

        # 计算模型预测值
        f_0 = self.first_order_models[self.variables[0]]['f0']  # 从模型中获取f0
        sum_fi_xe = 0.0

        # 对每个变量进行预测
        for variable in self.variables:
            if validation_data[variable] is not None and variable in self.first_order_models:
                # 获取验证数据点的变量值
                xe = np.array([[validation_data[variable]]])

                # 检查模型是否构建成功
                if self.first_order_models[variable]['model'] is None:
                    print(f"变量{variable}的模型不存在.跳过预测.")
                    return False

                # 如果是线性模型，则直接计算 fi(xe)
                if self.first_order_models[variable]['is_linear']:
                    # 获取初始样本点集
                    X, y, x0, f_i_L, f_i_R = self.construct_initial_sample_set(variable, target_function)
                    # 使用线性插值计算 fi(xe)
                    # 获取左端点和右端点
                    xi_values = self.yijie_data[self.yijie_data['变化变量'] == variable][variable].values
                    if len(xi_values) < 2:
                        # print(f"Less than 2 samples for variable {variable}. Skipping prediction.")
                        return False
                    x_L = xi_values[0]  # 左端点
                    x_R = xi_values[-1]  # 右端点
                    fi_xe = ((f_i_R - f_i_L) / (x_R - x_L)) * (xe - x0)
                    sum_fi_xe += fi_xe[0][0]  # 累加

                else:
                    # 如果是非线性模型，则使用 RBF 模型进行预测
                    model = self.first_order_models[variable]['model']
                    try:
                        # 调用 RBF 模型的 predict 方法进行预测
                        y_pred = model.predict(xe)
                        fi_xe = y_pred[0]  # 获取预测值
                        sum_fi_xe += fi_xe  # 累加
                    except Exception as e:
                        # print(f"Error predicting for variable {variable}: {e}")
                        return False
            else:
                # print(f"Validation data or model not available for variable {variable}.")
                return False

            # 计算 |f(x^e) - f_0 - sum(f_i(x_i^e))|
            diff = abs(f_xe - f_0 - sum_fi_xe)

            # 判断是否满足条件
            is_valid = diff < self.coupling_threshold

            print(f"一阶模型差值验证: |f(x^e) - f_0 - sum(f_i(x_i^e))| = {diff}, Valid = {is_valid}")
            return is_valid

    def determine_coupling_terms(self, target_function):
        """
        Determines the existence of first-order coupling terms between variables.

        Args:
            target_function (str): The name of the target function column.
        """
        coupled_terms = []
        for i in range(len(self.variables)):
            for j in range(i + 1, len(self.variables)):
                var_i = self.variables[i]
                var_j = self.variables[j]

                # 获取二阶样本中对应的变量组合数据
                combined_variable = f"{var_i},{var_j}"
                combined_data = self.erjie_data[self.erjie_data['变化变量'] == combined_variable]

                # 确保找到了对应的数据
                if not combined_data.empty:
                    # 获取目标函数值
                    f_0_ij = combined_data[target_function].values[0]

                    # 计算 f_i(x_i^e) 和 f_j(x_j^e)
                    fi_xe = self.predict_first_order(var_i, combined_data[var_i].values[0])
                    fj_xe = self.predict_first_order(var_j, combined_data[var_j].values[0])

                    # 计算 |f_0(x_i^e, x_j^e) - f_0 - f_i(x_i^e) - f_j(x_j^e)|
                    diff = abs(f_0_ij - self.f0 - fi_xe - fj_xe)/f_0_ij

                    # 判断是否存在耦合项
                    if diff >= self.coupling_threshold:
                        coupled_terms.append((var_i, var_j))
                        print(f"第六步：对变量{var_i} and {var_j}进行耦合判断. 计算结果: {diff}")
                    else:
                        print(f"变量 {var_i} and {var_j}不存在耦合. 计算结果: {diff}")
            else:
                print(f"没有耦合变量 {combined_variable}.")

        # 返回所有存在耦合项的变量组合
        return coupled_terms

    def predict_first_order(self, variable, x):
        """
        Predicts the response value for a given variable using the first-order RBF model.

        Args:
            variable (str): The name of the variable.
            x (float): The value of the variable at which to make the prediction.

        Returns:
            float: The predicted response value.
        """
        if variable not in self.first_order_models:
            # print(f"Error: First-order model not found for variable {variable}.")
            return 0.0

        model_data = self.first_order_models[variable]
        if model_data['is_linear']:
            # 如果是线性模型，使用线性插值进行预测
            X, y, x0, f_i_L, f_i_R = self.construct_initial_sample_set(variable, 'ex1')  # 使用 'ex1' 作为目标函数
            xi_values = self.yijie_data[self.yijie_data['变化变量'] == variable][variable].values
            x_L = xi_values[0]  # 左端点
            x_R = xi_values[-1]  # 右端点
            fi_x = ((f_i_R - f_i_L) / (x_R - x_L)) * (x - x0)
            return fi_x
        else:
            # 如果是非线性模型，使用 RBF 模型进行预测
            model = model_data['model']
            try:
                x = np.array([[x]])  # 将 x 转换为二维数组
                y_pred = model.predict(x)
                return y_pred[0]  # 返回预测值
            except Exception as e:
                # print(f"Error predicting for variable {variable}: {e}")
                return 0.0

    def verify_first_order_linearity(self, target_function):
        """
        Verifies the linearity of the first-order model for a given target function
        by calculating the relative error between the actual values and the
        first-order model predictions.

        Args:
            target_function (str): The name of the target function column.

        Returns:
            bool: True if the first-order model is considered linear (relative error < 0.01),
                  False otherwise.
        """
        # 确保一阶模型已经构建
        if not self.first_order_models:
            # print("Error: First-order models not yet built. Please run build_first_order_models first.")
            return False

        # 获取 first order model 数据
        first_order_data = self.yijie_data.copy()

        # 移除基准样本
        first_order_data = first_order_data[first_order_data['标号'] != 0]

        # 初始化总误差
        total_relative_error = 0.0
        num_samples = 0

        # 遍历first_order_data，进行计算
        for index, row in first_order_data.iterrows():
            variable = row['变化变量']
            actual_value = row[target_function]

            # 获取输入值 (确保它是正确的格式)
            x_value = row[variable]
            x_values = {variable: x_value}  # 创建字典
            input_value = np.array([[x_value]])

            #  计算一阶模型预测值，因为complete_hdmr_model 依赖于 所有的variable
            first_order_prediction = self.f0  # 初始化
            for model_variable in self.variables:
                if model_variable in self.first_order_models:
                    model_data = self.first_order_models[model_variable]
                    if model_data['model'] is not None:
                        if model_data['is_linear']:
                            # 如果是线性模型，使用线性插值计算 fi(x_i)
                            X, y, x0, f_i_L, f_i_R = self.construct_initial_sample_set(model_variable, target_function)
                            # 获取左端点和右端点
                            xi_values = self.yijie_data[self.yijie_data['变化变量'] == model_variable][
                                model_variable].values
                            if len(xi_values) < 2:
                                # print(f"Less than 2 samples for variable {model_variable}. Skipping prediction.")
                                continue
                            x_L = xi_values[0]  # 左端点
                            x_R = xi_values[-1]  # 右端点
                            fi_xe = ((f_i_R - f_i_L) / (x_R - x_L)) * (input_value - x0)
                            first_order_prediction += fi_xe[0][0]
                        else:
                            # 如果是非线性模型，使用 RBF 模型进行预测
                            y_pred = model_data['model'].predict(input_value)
                            first_order_prediction += y_pred[0]
                    else:
                        print(f"对于变量{model_variable}没有发现对应模型. Skipping.")
                else:
                    # print(f"Variable {model_variable} not found in first-order models. Skipping.")
                    print(f"Skipping.")
                    # 计算相对误差 (避免除以零)
            if actual_value != 0:
                relative_error = abs((actual_value - first_order_prediction) / actual_value) # (y-y')/y<0.01 y-y'
                total_relative_error += relative_error
                num_samples += 1
            else:
                # print(f"Warning: Actual value is zero for sample {index}. Skipping relative error calculation.")
                print(f"跳过.")

        # 计算平均相对误差
        if num_samples > 0:
            average_relative_error = total_relative_error / num_samples
            print(f"对于目标函数的平均相对误差： {target_function}: {average_relative_error:.4f}")
        else:
            # print(f"No valid samples for calculating relative error for Target Function {target_function}.")
            return False

        # 判断是否满足线性条件
        is_linear = average_relative_error < 0.01

        if is_linear:
            print(f"一阶模型 {target_function} 是线性模型.")
        else:
            print(f"一阶模型 {target_function} 是非线性模型.")

        return is_linear

    def build_second_order_models(self, target_function, coupled_terms):
        """
        Builds RBF-HDMR models for first-order coupled terms f_ij(x_i, x_j).
        It also checks if the first-order model is linear enough before building
        second-order models. If the first-order model is linear enough, the
        building of second-order models will be skipped.

        Args:
            target_function (str): The name of the target function column.
            coupled_terms (list): A list of tuples, where each tuple contains the names of two coupled variables (e.g., [('V1', 'V2'), ('V1', 'V3')]).
        """
        self.second_order_models = {}  # 初始化二阶模型
        for var_i, var_j in coupled_terms:
            print(f"对于变量 {var_i} and {var_j}构建二阶模型")

            # 定义非耦合项代理模型所用样本点集
            S_i = self.yijie_data[self.yijie_data['变化变量'] == var_i][var_i].values
            S_j = self.yijie_data[self.yijie_data['变化变量'] == var_j][var_j].values

            # 确保 S_i 和 S_j 不是空集
            if S_i.size == 0 or S_j.size == 0:
                # print(f"Skipping {var_i} and {var_j} due to empty sample sets.")
                continue

            # 获取基准样本值
            x_i_0 = self.yijie_data[self.yijie_data['标号'] == 0][var_i].values[0]
            x_j_0 = self.yijie_data[self.yijie_data['标号'] == 0][var_j].values[0]

            # 创建备选样本集 S_ij^a
            S_ij_a = np.array([[x_i, x_j] for x_i in S_i[1:] for x_j in S_j[1:]])  # 移除基准样本

            # 确保 S_ij_a 不是空集
            if S_ij_a.size == 0:
                # print(f"Skipping {var_i} and {var_j} due to empty candidate sample set.")
                continue

            # 创建初始样本集 S_ij^0（这里使用备选样本集中的前几个样本）
            num_initial_samples = min(5, len(S_ij_a))  # 限制初始样本数量
            initial_indices = np.random.choice(len(S_ij_a), num_initial_samples, replace=False)
            S_ij_0 = S_ij_a[initial_indices]
            remaining_indices = np.delete(np.arange(len(S_ij_a)), initial_indices)
            S_ij_remaining = S_ij_a[remaining_indices]

            # 计算初始样本集 S_ij^0 的响应值
            y_ij_0 = np.array([self.calculate_f_ij(var_i, x_i, var_j, x_j, target_function) for x_i, x_j in S_ij_0])

            # 构建初始代理模型
            rbf_model = self.build_rbf_model(S_ij_0, y_ij_0)

            # 设置收敛条件和最大迭代次数
            convergence_threshold = 1e-1
            max_iterations = 20

            # 循环添加样本点并更新模型
            S_ij_u = S_ij_0.copy()
            y_ij_u = y_ij_0.copy()
            converged_count = 0
            iteration = 0

            while converged_count < 3 and iteration < max_iterations:
                iteration += 1

                # 随机选择一个未使用的样本点进行测试
                if len(S_ij_remaining) == 0:
                    # print("No remaining samples to test. Stopping the update.")
                    break

                test_index = np.random.choice(len(S_ij_remaining))
                x_i_test, x_j_test = S_ij_remaining[test_index]
                S_ij_test = np.array([[x_i_test, x_j_test]])
                S_ij_remaining = np.delete(S_ij_remaining, test_index, axis=0)

                # 计算测试样本点的实际响应值
                y_ij_test = self.calculate_f_ij(var_i, x_i_test, var_j, x_j_test, target_function)

                # 使用代理模型预测测试样本点的响应值
                y_pred_test = rbf_model.predict(S_ij_test)

                # 判断是否满足收敛条件
                if abs(y_ij_test - y_pred_test[0]) < convergence_threshold:
                    converged_count += 1
                    print(
                        f"第{iteration}次迭代: 对于数据{x_i_test}, {x_j_test}收敛. 平均相对误差: {abs(y_ij_test - y_pred_test[0])}")
                else:
                    converged_count = 0
                    print(
                        f"第{iteration}次迭代: 对于数据{x_i_test}, {x_j_test}没有收敛. 平均相对误差: {abs(y_ij_test - y_pred_test[0])}")

                    # 将测试样本点加入更新样本集
                    S_ij_u = np.vstack((S_ij_u, S_ij_test))
                    y_ij_u = np.append(y_ij_u, y_ij_test)

                    # 更新代理模型
                    rbf_model = self.build_rbf_model(S_ij_u, y_ij_u)

            # 保存模型
            self.second_order_models[(var_i, var_j)] = rbf_model
            print(f"对变量{var_i} and {var_j}构建二阶模型.")

    def analyze_second_order_error(self, target_functions, coupled_terms):
        """
        Analyzes the relative error of the second-order models for each target function and coupled term.

        Args:
            target_functions (list): A list of target function names.
            coupled_terms (list): A list of tuples, where each tuple contains the names of two coupled variables.
        """
        error_data = {}  # 用于存储误差数据的字典

        for target_function in target_functions:
            error_data[target_function] = {}
            for var_i, var_j in coupled_terms:
                error_data[target_function][(var_i, var_j)] = []

                # 获取二阶样本中对应的变量组合数据
                combined_variable = f"{var_i},{var_j}"
                combined_data = self.erjie_data[self.erjie_data['变化变量'] == combined_variable]

                if not combined_data.empty:
                    # 获取目标函数实际值
                    f_values = combined_data[target_function].values

                    # 使用二阶模型预测
                    rbf_model = self.second_order_models.get((var_i, var_j))
                    if rbf_model is not None:
                        # 构建输入数据
                        input_values = combined_data[[var_i, var_j]].values

                        try:
                            # 预测值
                            y_pred = rbf_model.predict(input_values)

                            # 计算相对误差
                            relative_errors = np.abs((f_values - y_pred) / f_values)

                            # 存储误差数据
                            error_data[target_function][(var_i, var_j)] = relative_errors.tolist()

                            # 打印平均相对误差
                            mean_relative_error = np.mean(relative_errors)
                            print(
                                f"二阶代理模型: {target_function}, 耦合变量: {var_i},{var_j}, 平均相对误差: {mean_relative_error:.4f}")
                        except Exception as e:
                            print(
                                f"Error predicting for Target Function: {target_function}, Coupled Terms: {var_i},{var_j}: {e}")
                            error_data[target_function][(var_i, var_j)] = None
                    else:
                        print(
                            f"No RBF model found for Target Function: {target_function}, Coupled Terms: {var_i},{var_j}")
                        error_data[target_function][(var_i, var_j)] = None
                else:
                    print(
                        f"No data found for Target Function: {target_function}, Coupled Terms: {var_i},{var_j}")
                    error_data[target_function][(var_i, var_j)] = None

        # 将误差数据转换为 DataFrame 并打印
        for target_function, term_errors in error_data.items():
            print(f"\nTarget Function: {target_function}")
            for (var_i, var_j), errors in term_errors.items():
                if errors:
                    mean_error = np.mean(errors)
                    print(f"  Coupled Term: {var_i},{var_j}, Mean Relative Error: {mean_error:.4f}")
                else:
                    print(f"  Coupled Term: {var_i},{var_j}, No data or error occurred.")

    def calculate_f_ij(self, var_i, x_i, var_j, x_j, target_function):
        """
        Calculates the response value for the first-order coupling term f_ij(x_i, x_j).

        Args:
            var_i (str): The name of the first variable.
            x_i (float): The value of the first variable.
            var_j (str): The name of the second variable.
            x_j (float): The value of the second variable.
            target_function (str): The name of the target function column.

        Returns:
            float: The calculated response value.
        """
        # 获取二阶样本中对应的变量组合数据
        combined_variable = f"{var_i},{var_j}"
        combined_data = self.erjie_data[self.erjie_data['变化变量'] == combined_variable]

        # 确保找到了对应的数据
        if not combined_data.empty:
            # 获取目标函数值
            f_0_ij = combined_data[target_function].values[0]

            # 计算 f_i(x_i^e) 和 f_j(x_j^e)
            fi_xe = self.predict_first_order(var_i, x_i)
            fj_xe = self.predict_first_order(var_j, x_j)

            # 计算 f_ij(x_i, x_j) = f_0(x_i^e, x_j^e) - f_0 - f_i(x_i^e) - f_j(x_j^e)
            # f_ij = (f_0_ij - self.f0 - fi_xe - fj_xe)
            f_ij = abs((f_0_ij - self.f0 - fi_xe - fj_xe)/f_0_ij)
            return f_ij
        else:
            # print(f"No data found for variable combination {combined_variable}.")
            return 0.0

    def create_plots(self, target_function):
        """
        Creates plots for the first-order RBF-HDMR models.

        Args:
            target_function (str): The name of the target function column.
        """
        # 确保文件夹存在
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for variable, model_data in self.first_order_models.items():
            if model_data['model'] is not None:
                # 获取数据
                f_0 = self.first_order_models[variable]['f0']
                X, y, _, _, _ = self.construct_initial_sample_set(variable, target_function)
                x_plot = np.linspace(min(X), max(X), 100).reshape(-1, 1)  # 产生平滑的 x 值用于绘图, 并确保是二维数组
                y_pred = model_data['model'].predict(x_plot) # 获取预测值
                # 计算 MSE
                MSE = np.zeros_like(x_plot)
                for i, x in enumerate(x_plot):
                    MSE[i] = np.sqrt(np.diag(model_data['model'].kernel_(x.reshape(1, -1), X)))

                # 绘制 RBF 模型
                plt.figure()
                y_pred1 = gaussian_filter1d(y_pred, sigma=3)
                r2 = r2_score(y_pred, y_pred1)
                plt.scatter(X, y, label='Sample Points')
                plt.plot(x_plot, y_pred1, label=f'RBF Prediction ($R^2$={r2:.2f})')
                # plt.fill_between(x_plot.flatten(),
                #                  y_pred.flatten() - 1.96 * MSE.flatten(),
                #                  y_pred.flatten() + 1.96 * MSE.flatten(),
                #                  alpha=0.2, label='95% Confidence Interval')
                plt.xlabel(variable)
                plt.ylabel(target_function)
                plt.title(f'RBF Model for {variable} - {target_function}')
                plt.legend()
                plt.grid(True)

                # 保存图形
                filename = f'RBF_model_{variable}_{target_function}.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath)
                plt.close()
                print(f"Plot saved to {filepath}")
            else:
                print(f"Model is None for {variable}, skipping plot creation.")

    def create_summary_plot(self, target_functions):
        """
        Creates a 2x5 summary plot showing the RBF model for each variable
        with respect to each target function, and annotates the R^2 score.

        Args:
            target_functions (list): A list of target function names.
        """
        num_rows = 2
        num_cols = 5
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))  # 调整 figsize

        # 确保 axes 是一个二维数组，即使只有一行
        if num_rows == 1 or num_cols == 1:
            axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes.reshape(num_rows, num_cols)

        # 确保文件夹存在
        output_dir = "plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plot_index = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if plot_index >= len(target_functions):
                    axes[i, j].axis('off')  # 如果子图数量超过了函数数量，则关闭坐标轴
                    continue

                target_function = target_functions[plot_index]
                self.calculate_f0(target_function)  # 确保计算了正确的 f0 值
                ax = axes[i, j]
                ax.set_title(f'Target: {target_function}', fontsize=12)  # 调整标题字体大小
                ax.tick_params(axis='both', labelsize=8)  # 调整坐标轴刻度标签字体大小

                for variable in self.variables:
                    if variable in self.first_order_models:
                        model_data = self.first_order_models[variable]
                        if model_data['model'] is not None:
                            # 获取数据
                            X, y, _, _, _ = self.construct_initial_sample_set(variable, target_function)
                            X_plot = np.linspace(min(X), max(X), 100).reshape(-1, 1)
                            y_pred = model_data['model'].predict(X_plot)

                            # 计算 R^2
                            y_true = model_data['model'].predict(X)
                            r2 = r2_score(y, y_true)
                            ax.plot(X_plot, y_pred, label=f'{variable} (R^2={r2:.2f})')  # 添加变量名称和 R^2

                ax.legend(loc='upper left', fontsize='small')  # 添加图例，并调整字体大小
                ax.grid(True)
                plot_index += 1

        plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
        # 保存图形
        filename = f'summary_plot.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()
        print(f"Summary plot saved to {filepath}")

    def create_second_order_plots(self, target_function):
        """
        Creates plots for the second-order RBF-HDMR models.
        """
        # 确保文件夹存在
        output_dir = "second_order_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for (var_i, var_j), model in self.second_order_models.items():
            if model is not None:
                # 获取数据
                S_i = self.yijie_data[self.yijie_data['变化变量'] == var_i][var_i].values
                S_j = self.yijie_data[self.yijie_data['变化变量'] == var_j][var_j].values

                # 创建备选样本集 S_ij^a
                S_ij_a = np.array([[x_i, x_j] for x_i in S_i[1:] for x_j in S_j[1:]])  # 移除基准样本

                # 创建网格用于绘图
                num_points = 50
                x_i_plot = np.linspace(min(S_ij_a[:, 0]), max(S_ij_a[:, 0]), num_points)
                x_j_plot = np.linspace(min(S_ij_a[:, 1]), max(S_ij_a[:, 1]), num_points)
                X_i, X_j = np.meshgrid(x_i_plot, x_j_plot)
                X_plot = np.vstack((X_i.flatten(), X_j.flatten())).T

                # 使用模型进行预测
                y_pred = model.predict(X_plot)

                # 绘制三维曲面图
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(X_i, X_j, y_pred.reshape(num_points, num_points), cmap='viridis')

                ax.set_xlabel(var_i)
                ax.set_ylabel(var_j)
                ax.set_zlabel(target_function)
                ax.set_title(f'Second-Order RBF Model for {var_i} and {var_j} - {target_function}')

                # 保存图形
                filename = f'second_order_model_{var_i}_{var_j}_{target_function}.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath)
                plt.close()
                print(f"Second-order plot saved to {filepath}")
            else:
                print(f"Second-order model is None for {var_i} and {var_j}, skipping plot creation.")

    def create_summary_plots_per_target(self, target_functions):
        """
        Creates a summary plot for each target function, with 2x5 subplots
        showing the RBF model for each variable, and annotates the R^2 score.

        Args:
            target_functions (list): A list of target function names.
        """
        for target_function in target_functions:
            num_rows = 2
            num_cols = 5
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 8))

            # 确保 axes 是一个二维数组，即使只有一行
            if num_rows == 1 or num_cols == 1:
                axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes.reshape(num_rows, num_cols)

            # 确保文件夹存在
            output_dir = "plots"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 遍历所有变量
            var_index = 0
            for i in range(num_rows):
                for j in range(num_cols):
                    if var_index >= len(self.variables):
                        axes[i, j].axis('off')  # 如果子图数量超过了变量数量，则关闭坐标轴
                        continue

                    variable = self.variables[var_index]
                    ax = axes[i, j]
                    ax.set_title(f'Var: {variable}', fontsize=12)
                    ax.tick_params(axis='both', labelsize=8)

                    if variable in self.first_order_models:
                        model_data = self.first_order_models[variable]
                        if model_data['model'] is not None:
                            # 获取数据 (使用构建模型时使用的数据)
                            X, y, _, _, _ = self.construct_initial_sample_set(variable, target_function)
                            X_plot = np.linspace(min(X), max(X), 100).reshape(-1, 1)
                            y_pred = model_data['model'].predict(X_plot)

                            # 计算 R^2
                            y_true = model_data['model'].predict(X)
                            r2 = r2_score(y, y_true)

                            # 绘制 RBF 模型
                            ax.scatter(X, y, label='Sample Points', s=10)  # 绘制样本点
                            ax.plot(X_plot, y_pred, label=f'RBF (R^2={r2:.2f})')
                            ax.legend(loc='upper left', fontsize='small')
                            ax.grid(True)
                        else:
                            ax.text(0.5, 0.5, 'Model Building Failed', ha='center', va='center', fontsize=10,
                                    color='red')
                    else:
                        ax.text(0.5, 0.5, 'Model Not Available', ha='center', va='center', fontsize=10, color='red')

                    var_index += 1

            plt.suptitle(f'Target Function: {target_function}', fontsize=16)  # 添加总标题
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整子图，为总标题留出空间

            # 保存图形
            filename = f'summary_plot_{target_function}.png'
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath)
            plt.close()
            print(f"Summary plot for {target_function} saved to {filepath}")

    def save_f0(self, target_function, f0):
        """
        Saves the baseline response value f0 to a file.

        Args:
            target_function (str): The name of the target function column.
            f0 (float): The baseline response value.
        """
        # 使用目标函数名称创建一个子文件夹
        target_folder = os.path.join(self.save_model_path, target_function)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        filename = 'f0.pkl'
        filepath = os.path.join(target_folder, filename)

        try:
            with open(filepath, 'wb') as file:
                pickle.dump(f0, file)
            print(f"f0 value for {target_function} saved to {filepath}")
        except Exception as e:
            print(f"Error saving f0 value for {target_function}: {e}")

    def load_f0(self, target_function):
        """
        Loads the baseline response value f0 from a file.

        Args:
            target_function (str): The name of the target function column.
        """
        # 构建完整的子文件夹路径
        target_folder = os.path.join(self.save_model_path, target_function)
        filename = 'f0.pkl'
        filepath = os.path.join(target_folder, filename)

        try:
            with open(filepath, 'rb') as file:
                self.f0 = pickle.load(file)
            print(f"f0 value for {target_function} loaded from {filepath}")
        except FileNotFoundError:
            print(f"f0 file not found for {target_function} at {filepath}. Skipping.")
        except Exception as e:
            print(f"Error loading f0 value for {target_function}: {e}")

    def save_first_order_models(self):
        """
        Saves the parameters of the first-order RBF-HDMR models.
        """
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        for target_function in self.target_functions:
            # 使用目标函数名称创建一个子文件夹
            target_folder = os.path.join(self.save_model_path, target_function)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            for variable, model_data in self.first_order_models.items():
                if model_data['model'] is not None:
                    # 模型文件名
                    filename = f'RBF_model_{variable}.pkl'
                    filepath = os.path.join(target_folder, filename)

                    try:
                        # 保存模型
                        with open(filepath, 'wb') as file:
                            pickle.dump(model_data, file)

                        print(f"Model for {variable} - {target_function} saved to {filepath}")
                    except Exception as e:
                        print(f"Error saving model for {variable} - {target_function}: {e}")
                else:
                    print(f"Model is None for {variable} - {target_function}, skipping save.")

    def save_second_order_models(self):
        """
        Saves the parameters of the second-order RBF-HDMR models.
        """
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        # 使用目标函数名称创建一个子文件夹
        for target_function in self.target_functions:
            target_folder = os.path.join(self.save_model_path, target_function)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            for (var_i, var_j), model in self.second_order_models.items():
                if model is not None:
                    # 模型文件名
                    filename = f'RBF_model_{var_i}_{var_j}.pkl'
                    filepath = os.path.join(target_folder, filename)
                    try:
                        # 保存模型
                        with open(filepath, 'wb') as file:
                            pickle.dump(model, file)
                        print(f"Second-order model for {var_i} and {var_j} - {target_function} saved to {filepath}")
                    except Exception as e:
                        print(f"Error saving second-order model for {var_i} and {var_j} - {target_function}: {e}")
                else:
                    print(f"Second-order model is None for {var_i} and {var_j} - {target_function}, skipping save.")

    def load_first_order_models(self, target_function):
        """
        Loads the parameters of the first-order RBF-HDMR models.
        """
        self.first_order_models = {}  # 清空现有模型

        # 构建完整的子文件夹路径
        target_folder = os.path.join(self.save_model_path, target_function)

        # 加载 f0 值
        self.load_f0(target_function)

        for variable in self.variables:
            # 构建完整的文件路径
            filename = f'RBF_model_{variable}.pkl'
            filepath = os.path.join(target_folder, filename)

            try:
                # 尝试加载模型
                with open(filepath, 'rb') as file:
                    model_data = pickle.load(file)
                    self.first_order_models[variable] = model_data
                    print(f"Model for {variable} - {target_function} loaded from {filepath}")
            except FileNotFoundError:
                print(f"Model file not found for {variable} - {target_function} at {filepath}. Skipping.")
            except Exception as e:
                print(f"Error loading model for {variable} - {target_function}: {e}")

    def load_second_order_models(self, target_function):
        """
        Loads the parameters of the second-order RBF-HDMR models.
        """
        self.second_order_models = {}  # 清空现有模型

        # 构建完整的子文件夹路径
        target_folder = os.path.join(self.save_model_path, target_function)

        # 加载二阶模型
        for var_i in self.variables:
            for var_j in self.variables:
                if var_i == var_j:
                    continue

                # 构建变量对，并进行排序，保证模型名称的顺序一致
                vars = sorted([var_i, var_j])
                var_i, var_j = vars[0], vars[1]
                model_name = f'RBF_model_{var_i}_{var_j}.pkl'

                # 构建完整的文件路径
                filepath = os.path.join(target_folder, model_name)
                try:
                    # 尝试加载模型
                    with open(filepath, 'rb') as file:
                        model = pickle.load(file)
                        self.second_order_models[(var_i, var_j)] = model
                        print(f"Second-order model for {var_i},{var_j} - {target_function} loaded from {filepath}")
                except FileNotFoundError:
                    continue
                except Exception as e:
                    continue

    def build_complete_RBF_hdmr_model(self, target_function):
        """
        Combines the constant term, first-order terms, and second-order coupled terms
        to form the complete RBF-HDMR model.

        Args:
            target_function (str): The name of the target function column.

        Returns:
            function: A callable function representing the complete RBF-HDMR model.
        """
        # 确保基准值 f0 和一阶模型已经计算并存储
        if self.f0 is None or not self.first_order_models:
            print("Error: First-order models or f0 not yet calculated. Please run build_first_order_models first.")
            return None

        def complete_hdmr_model(x_values):
            """
            The complete RBF-HDMR model function.

            Args:
                x_values (dict): A dictionary where keys are variable names (e.g., 'V1')
                                 and values are the corresponding input values.

            Returns:
                float: The predicted response value.
            """
            # 初始化总响应值为常数项 f0
            total_response = self.f0

            # 累加一阶项的预测值
            for variable in self.variables:
                if variable in self.first_order_models:
                    model_data = self.first_order_models[variable]
                    if model_data['model'] is not None:
                        input_value = np.array([[x_values.get(variable)]])
                        if model_data['is_linear']:
                            # 如果是线性模型，使用线性插值计算 fi(x_i)
                            # 获取初始样本点集
                            X, y, x0, f_i_L, f_i_R = self.construct_initial_sample_set(variable, target_function)
                            # 获取左端点和右端点
                            xi_values = self.yijie_data[self.yijie_data['变化变量'] == variable][variable].values
                            if len(xi_values) < 2:
                                print(f"Less than 2 samples for variable {variable}. Skipping prediction.")
                                return False
                            x_L = xi_values[0]  # 左端点
                            x_R = xi_values[-1]  # 右端点
                            fi_xe = ((f_i_R - f_i_L) / (x_R - x0)) * (input_value - x0)
                            total_response += fi_xe[0][0]
                        else:
                            # 如果是非线性模型，使用 RBF 模型进行预测
                            try:
                                # 将输入值转换为二维数组
                                input_value = np.array(input_value)
                                y_pred = model_data['model'].predict(input_value)
                                total_response += y_pred[0]
                            except Exception as e:
                                print(f"Error predicting first-order term for {variable}: {e}")

                    else:
                        print(f"No model found for variable {variable}. Skipping.")
                else:
                    print(f"Variable {variable} not found in first_order_models. Skipping.")

            # 累加二阶耦合项的预测值
            for (var_i, var_j), model in self.second_order_models.items():
                if model is not None:
                    try:
                        input_values = np.array([[x_values.get(var_i), x_values.get(var_j)]])
                        f_ij_x = model.predict(input_values)
                        total_response += f_ij_x[0]
                    except Exception as e:
                        print(f"Error predicting second-order term for {var_i},{var_j}: {e}")
                        # 在出错时选择不累加该项，或根据需求处理
                        pass
            else:
                # 如果模型为 None，则该耦合项为 0
                pass

            return total_response

        return complete_hdmr_model

    def save_complete_model(self):
        """
        保存构建好的完整的 RBF-HDMR 模型。
        """
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        # 使用目标函数名称创建一个子文件夹
        for target_function in self.target_functions:
            target_folder = os.path.join(self.save_model_path, target_function)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            try:
                # 构建完整模型
                complete_model = self.build_complete_RBF_hdmr_model(target_function)

                if complete_model:
                    filename = f'complete_model_{target_function}.pkl'
                    filepath = os.path.join(target_folder, filename)

                    with open(filepath, 'wb') as file:
                        pickle.dump({
                            'f0': self.f0,
                            'first_order_models': self.first_order_models,
                            'second_order_models': self.second_order_models,
                            'model_function': complete_model  # 保存模型函数本身
                        }, file)
                    print(f"complete Model saved to {filepath}")

            except Exception as e:
                print(f"Error saving complete model for {target_function}: {e}")

    def load_complete_model(self):
        """
        加载完整的 RBF-HDMR 模型。
        """
        # 使用目标函数名称创建一个子文件夹
        for target_function in self.target_functions:

            target_folder = os.path.join(self.save_model_path, target_function)

            filename = f'complete_model_{target_function}.pkl'
            filepath = os.path.join(target_folder, filename)
            try:
                with open(filepath, 'rb') as file:
                    model_data = pickle.load(file)
                    # 加载模型的各个部分
                    self.f0 = model_data['f0']
                    self.first_order_models = model_data['first_order_models']
                    self.second_order_models = model_data['second_order_models']
                    self.complete_model_function = model_data['model_function']

            except Exception as e:
                print(f"Error loading complete model for {target_function}: {e}")

    def optimize_model(self, target_function_type="obj_fun1"):
        """
        Optimizes the complete RBF-HDMR model using Particle Swarm Optimization (PSO)
        to find the values of V1-V10 that minimize the specified objective function,
        and then uses NSGA-II to select Pareto front points from all solutions.

        ... (Docstring remains the same) ...
        """

        # 定义变量的范围 (保持不变)
        vars_lb = np.array([0.0001, 20, 20, 45, 45, 0.0001, 15, 0.0001, 15, 40])
        vars_ub = np.array([5, 40, 30, 50, 65, 1, 25, 1, 25, 65])
        vars_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10']
        dim = len(vars_names)

        # 定义目标函数 (保持不变)
        def objective_function(x, target_function):
            # ... (function content remains the same) ...
            x_values = dict(zip(vars_names, x))
            complete_model = self.build_complete_RBF_hdmr_model(target_function)
            if complete_model is None:
                raise ValueError("Complete HDMR model could not be built.")
            try:
                obj_val = complete_model(x_values)
            except Exception as e:
                print(f"Error during model evaluation: {e}")
                obj_val = np.inf
            return obj_val

        # 目标函数1 & 2 (保持不变)
        def evalVars1(x):
            # ... (function content remains the same) ...
            ex1 = objective_function(x, 'ex1')
            ex2 = objective_function(x, 'ex2')
            ex3 = objective_function(x, 'ex3')
            return ex1 + ex2 + ex3

        def evalVars2(x):
            # ... (function content remains the same) ...
            ex4 = objective_function(x, 'ex4')
            head = objective_function(x, 'head')
            return ex4 + head

        # 优化问题设置 (保持不变)
        if target_function_type == "obj_fun1":
            evalVars = evalVars1
        elif target_function_type == "obj_fun2":
            evalVars = evalVars2
        else:
            raise ValueError("Invalid target_function_type. Must be 'obj_fun1' or 'obj_fun2'.")

        # PSO 参数 (保持不变)
        n_particles = 10
        max_iter = 5
        w = 0.7
        c1 = 1.5
        c2 = 1.5

        # 初始化粒子 (保持不变)
        particles = np.random.uniform(vars_lb, vars_ub, size=(n_particles, dim))
        velocities = np.random.uniform(-1, 1, size=(n_particles, dim))

        # 初始化个体最佳位置和全局最佳位置 (保持不变)
        pbest_positions = particles.copy()
        pbest_values = np.array([evalVars(particle) for particle in particles])
        gbest_position = particles[np.argmin(pbest_values)]
        gbest_value = np.min(pbest_values)

        # 收集所有可行解及其目标函数值
        all_vars = []
        all_obj_vals = []

        # 迭代 (保持不变)
        for i in range(max_iter):
            # ... (PSO iteration logic remains the same) ...
            r1 = np.random.rand(n_particles, dim)
            r2 = np.random.rand(n_particles, dim)
            velocities = w * velocities + c1 * r1 * (pbest_positions - particles) + c2 * r2 * (
                    gbest_position - particles)
            particles = particles + velocities
            particles = np.clip(particles, vars_lb, vars_ub)
            values = np.array([evalVars(particle) for particle in particles])

            for particle, value in zip(particles, values):
                all_vars.append(particle)
                all_obj_vals.append(value)

            mask = values < pbest_values
            pbest_positions[mask] = particles[mask]
            pbest_values[mask] = values[mask]

            if np.min(pbest_values) < gbest_value:
                gbest_position = pbest_positions[np.argmin(pbest_values)]
                gbest_value = np.min(pbest_values)

            print(f"Iteration {i + 1}: Best objective function value = {gbest_value}")

        # 提取结果 (保持不变)
        best_vars = dict(zip(vars_names, gbest_position))
        best_obj_val = gbest_value

        print("Best variable values:", best_vars)
        print("Best objective function value:", best_obj_val)

        # 寻找 Pareto 前沿并选取点
        def find_pareto_front_and_select_points(all_vars, all_obj_vals, n_points=10):  # n_points
            # ... (Docstring remains the same) ...

            # 转换为 numpy 数组
            all_vars = np.array(all_vars)
            all_obj_vals = np.array(all_obj_vals)

            # 确保 all_obj_vals 是一维数组
            if all_obj_vals.ndim != 1:
                raise ValueError("all_obj_vals must be a 1D numpy array.")

            # 创建 Problem 类
            # 关键修改：继承 ElementwiseProblem
            class MyProblem(ElementwiseProblem):
                def __init__(self, all_vars, all_obj_vals):
                    # 关键修改：移除 elementwise_evaluation=True
                    super().__init__(n_var=all_vars.shape[1],
                                     n_obj=1,
                                     n_constr=0,
                                     xl=vars_lb,
                                     xu=vars_ub)
                    self.all_vars = all_vars
                    self.all_obj_vals = all_obj_vals

                def _evaluate(self, x, out, *args, **kwargs):
                    # 计算欧氏距离
                    # 确保 x 是 1D 数组，以便广播正常工作
                    x = np.atleast_1d(x)
                    distances = np.sum((self.all_vars - x) ** 2, axis=1)
                    # 找到最近点的索引
                    index = np.argmin(distances)

                    # 提取对应的函数值
                    F = self.all_obj_vals[index]
                    out["F"] = np.array([F])

            problem = MyProblem(all_vars, all_obj_vals)

            # 使用NSGA2算法寻找Pareto前沿
            algorithm = NSGA2(pop_size=min(len(all_vars), 100))

            # 定义终止条件
            termination = get_termination("n_gen", 5)

            # 优化
            res = minimize(problem,
                           algorithm,
                           termination,
                           seed=1,
                           verbose=False)

            # 获取 Pareto 前沿上的解和目标函数值
            # 检查 res.X 和 res.F 是否存在且不为空
            if res.X is None or res.F is None or len(res.X) == 0:
                print("Warning: No Pareto front solutions found.")
                return [], np.array([])

            pareto_vars_array = res.X
            pareto_obj_vals = res.F.flatten()
            print(len(pareto_vars_array))
            # 在 Pareto 前沿上选择点 (保持不变)
            if n_points > len(pareto_vars_array):
                print(
                    f"Warning: n_points is greater than the number of Pareto front solutions. Reducing n_points to {len(pareto_vars_array)}.")
                n_points = len(pareto_vars_array)

            # 如果 pareto_vars_array 只有一行，np.linspace 可能会有问题，需要处理
            n_pareto_points = pareto_vars_array.shape[0]
            if len(pareto_vars_array) <= 1:
                selected_indices = [0]
            else:
                selected_indices = np.linspace(0, len(pareto_vars_array) - 1, n_points, dtype=int)

            # 确保 selected_indices 不超出范围
            # selected_indices = np.linspace(0, n_pareto_points - 1, n_points, dtype=int)
            # selected_indices = np.clip(selected_indices, 0, n_pareto_points - 1)
            # pareto_vars_array = pareto_vars_array.reshape(-1, dim)
            pareto_vars = [dict(zip(vars_names, pareto_vars_array[i])) for i in selected_indices]
            pareto_obj_vals = pareto_obj_vals[selected_indices]

            print("Selected Pareto front variable values:", pareto_vars)
            print("Selected Pareto front objective function values:", pareto_obj_vals)

            return pareto_vars, pareto_obj_vals

        pareto_vars, pareto_obj_vals = find_pareto_front_and_select_points(all_vars, all_obj_vals)

        # 绘制 Pareto 前沿图 (保持不变)
        if len(pareto_obj_vals) > 0:
            plt.figure(figsize=(8, 6))
            plt.scatter(pareto_obj_vals, np.zeros_like(pareto_obj_vals), c='red', marker='o',
                        label='Pareto Front Solutions')
            plt.xlabel('Objective Function Value')
            plt.title('Pareto Front Solutions')
            plt.legend()
            plt.grid(True)
            plt.show()

        return best_vars, best_obj_val, pareto_vars, pareto_obj_vals

    def load_test_data(self, test_file):
        """
        Loads test data from the specified CSV file.

        Args:
            test_file (str): Filename for the test data (CSV).

        Returns:
            pandas.DataFrame: The test data.
        """
        try:
            file_path = os.path.join(self.path, test_file)
            test_data = pd.read_csv(file_path)

            # 排除“变化变量”这一列
            if '变量（随便写了两个）' in test_data.columns:
                test_data = test_data.drop('变量（随便写了两个）', axis=1)
            if '10个都变-样本' in test_data.columns:
                test_data = test_data.drop('10个都变-样本', axis=1)
            print(f"Test data loaded successfully from {file_path}.")
            return test_data
        except FileNotFoundError:
            print(f"Error: Test data file not found at {file_path}.")
            exit()
        except Exception as e:
            print(f"Error loading test data: {e}")
            exit()

    def run(self):
        """
        Runs the RBF-HDMR modeling process based on the specified mode.
        """
        if self.mode == 'first':
            # 构建和保存一阶 RBF-HDMR 模型
            self.load_first_order_data()
            for target_function in self.target_functions:
                print(f"Building first-order models for target function: {target_function}")
                self.build_first_order_models(target_function)
                is_valid = self.validate_first_order_model(target_function)
                if is_valid:
                    print("First-order RBF-HDMR model is valid.")
                else:
                    print("First-order RBF-HDMR model is not valid.")
                self.create_plots(target_function)  # 创建模型图
            self.save_first_order_models()  # 保存模型
            # 创建 Summary Plot
            self.create_summary_plot(self.target_functions)
            self.create_summary_plots_per_target(self.target_functions)
        elif self.mode == 'second':
            # 构建二阶 RBF-HDMR 模型
            self.load_first_order_data()
            self.load_second_order_data()
            for target_function in self.target_functions:
                print(f"Building second-order models for target function: {target_function}")
                # 验证一阶模型线性度,如果是一阶模型，那么直接跳出，不进行二阶模型的构建
                # 加载一阶模型参数
                self.load_first_order_models(target_function)
                # 确定耦合项
                self.calculate_f0(target_function)
                is_linear = self.verify_first_order_linearity(target_function)
                if is_linear:
                    print(f"Skipping second-order model building for {target_function} due to linearity.")
                    self.second_order_models = {}  # 清空二阶模型
                    continue  # 直接返回
                coupled_terms = self.determine_coupling_terms(target_function)
                # 构建二阶模型
                self.build_second_order_models(target_function, coupled_terms)
                # 构建完整模型
                # complete_model = self.build_complete_RBF_hdmr_model(target_function)

                # 分析二阶模型的误差
                # self.analyze_second_order_error(self.target_functions, coupled_terms)
            best_vars, best_obj_val, _, _ = self.optimize_model(target_function_type="obj_fun1")

            print("Optimization completed.")
        elif self.mode == 'third':
            # 加载一阶和二阶模型，构建完整的 RBF-HDMR 模型
            self.load_first_order_data()
            self.load_second_order_data()
            # 加载测试数据
            test_data = self.load_test_data(self.test_file)
            for target_function in self.target_functions:
                # print(f"Loading models and building complete model for target function: {target_function}")
                # 加载一阶模型
                self.load_first_order_models(target_function)

                # 加载二阶模型
                self.load_second_order_models(target_function)
                # 构建完整模型
                complete_model = self.build_complete_RBF_hdmr_model(target_function)

                if complete_model:
                    print(f"Complete RBF-HDMR model built successfully for {target_function}.")
                    # 在这里可以对 complete_model 进行测试、验证或保存

                    # 使用测试数据进行预测和对比
                    predictions = []
                    actual_values = []

                    for index, row in test_data.iterrows():
                        # 提取测试数据的变量值
                        x_values = row.to_dict()

                        try:
                            # 使用构建好的完整模型进行预测
                            prediction = complete_model(x_values)
                            predictions.append(prediction)
                            actual_values.append(row[target_function])  # 假设测试数据包含目标函数列
                        except Exception as e:
                            print(f"Error during prediction for sample {index}: {e}")
                            predictions.append(None)
                            actual_values.append(None)

                    # 对比预测值和真实值
                    valid_predictions = [p for p in predictions if p is not None]
                    valid_actual_values = [a for a, p in zip(actual_values, predictions) if p is not None]

                    if valid_predictions:
                        # 计算误差指标，例如均方误差 (MSE)
                        mse = np.mean((np.array(valid_predictions) - np.array(valid_actual_values)) ** 2)
                        #R2
                        print(f"Mean Squared Error (MSE) for {target_function}: {mse}")

                        # 可以将预测值和真实值保存到文件，或者进行可视化分析
                    else:
                        print(f"No valid predictions for {target_function}.")

                else:
                    print(f"Failed to build complete RBF-HDMR model for {target_function}.")
        else:
            print("Error: Invalid mode specified. Choose 'first' 'second' or 'third.")


if __name__ == "__main__":
    sys.stdout = Logger()
    parser = argparse.ArgumentParser(description="Build RBF-HDMR surrogate model.")
    parser.add_argument("--mode", type=str, default='second',  # 选择构建几阶模型
                        help="Mode: 'first' for first-order, 'second' for second-order modeling.")
    parser.add_argument("--path", type=str, default='./dataset', help="Path to the data files.")
    parser.add_argument("--yijie_file", type=str, default='yijie2.csv', help="Filename for the first-order data (CSV).")
    parser.add_argument("--erjie_file", type=str, default='erjie.csv', help="Filename for the second-order data (CSV).")
    parser.add_argument("--test_file", type=str, default='test.csv', help="Filename for the second-order data (CSV).")
    parser.add_argument("--save_model", type=str, default='./checkpoints/',
                        help="Path to save the trained RBF-HDMR model parameters.")
    args = parser.parse_args()

    # 实例化并运行 RBF-HDMR 模型
    hdmr_RBF = HDMR_RBF(args.mode, args.path, args.yijie_file, args.erjie_file, args.save_model, args.test_file)
    hdmr_RBF.run()

    # 关闭日志文件（重要）
    sys.stdout = original_stdout  # 恢复标准输出
    log_file.close()
