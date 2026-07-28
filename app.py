# -*- coding: utf-8 -*-
"""
第8天：Flask 项目强化 — 电商数据看板 API 化
"""

import io
import csv

from functools import wraps
from pathlib import Path

from flask import (
    Flask, flash, jsonify, redirect,
    render_template, request, session, url_for, send_file,
)

from services.data_service import (
    load_dashboard_data,
    get_api_metrics,      # 第8天新增
    get_api_categories,   # 第8天新增
)
from services.qa_service import answer_question


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config["SECRET_KEY"] = "day08-classroom-demo-key"


# ---------------------------------------------------------------------------
# 装饰器
# ---------------------------------------------------------------------------

def login_required(view):
    """第7天原有：Session 登录检查，未登录跳转 /login"""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "username" not in session:
            flash("请先登录后再访问数据看板。", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped_view


def api_login_required(view):
    """第8天新增：API 登录检查，未登录返回 JSON 错误（而非重定向）"""
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "username" not in session:
            # TODO 8-3：统一错误响应 — 错误格式含 code 和 error，状态码 401
            return jsonify({"ok": False, "error": "未登录，请先访问 /login 登录。"}), 401
        return view(*args, **kwargs)
    return wrapped_view


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    """第8天新增：健康检查端点，无需登录，返回 JSON。"""
    return jsonify({"ok": True, "status": "healthy"})


# ---------------------------------------------------------------------------
# 第7天原有页面路由（保持兼容）
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("dashboard") if "username" in session else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # request.form 用于接收 HTML 表单提交
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == "student" and password == "day07":
            session["username"] = username
            flash("登录成功，欢迎进入电商用户分析系统。", "success")
            return redirect(url_for("dashboard"))
        flash("账号或密码错误。演示账号：student / day07", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("你已安全退出。", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    # request.args.get() 用于读取 URL 查询参数 ?category=Fashion
    category = request.args.get("category", "全部")
    dashboard_data = load_dashboard_data(BASE_DIR, category)
    return render_template(
        "dashboard.html",
        username=session["username"],
        selected_category=category,
        **dashboard_data,
    )


@app.route("/assistant")
@login_required
def assistant():
    return render_template("assistant.html", username=session["username"])


@app.route("/api/ask", methods=["POST"])
@login_required
def ask():
    # request.get_json() 用于读取 JSON 请求体
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"ok": False, "answer": "请输入一个与项目数据有关的问题。"}), 400
    return jsonify({"ok": True, "answer": answer_question(BASE_DIR, question)})


# ---------------------------------------------------------------------------
# TODO 8-1：指标 API
#   - 登录后返回 JSON 格式的指标数据
#   - 数据必须来自 data_service，不能硬编码
# ---------------------------------------------------------------------------
@app.route("/api/metrics")
@api_login_required
def api_metrics():
    """返回核心指标 JSON：总用户数、流失用户、流失率、平均订单数。"""
    # TODO 8-4：get_api_metrics 返回的是普通 Python dict，jsonify() 可直接序列化
    metrics = get_api_metrics(BASE_DIR)
    return jsonify({"ok": True, "metrics": metrics})


# ---------------------------------------------------------------------------
# TODO 8-2：品类 API
#   - 支持 ?category=Fashion 参数筛选
#   - category 必须真正进入筛选逻辑（not hardcoded）
# ---------------------------------------------------------------------------
@app.route("/api/categories")
@api_login_required
def api_categories():
    """返回品类数据 JSON，支持 ?category=X 筛选。"""
    category = request.args.get("category")
    # category 参数直接传入数据服务层，由 data_service 执行筛选逻辑
    data = get_api_categories(BASE_DIR, category)
    return jsonify({"ok": True, "category": category or "全部", "rows": data})


# ---------------------------------------------------------------------------
# TODO 8-3：统一错误响应
#   错误格式：{"ok": false, "error": "错误描述"}
#   不可以用 200 状态码伪装失败请求
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"ok": False, "error": "请求参数有误。"}), 400


@app.errorhandler(404)
def page_not_found(_error):
    # 根据请求头决定返回 HTML 页面还是 JSON
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "资源不存在。"}), 404
    return render_template("404.html"), 404


@app.errorhandler(405)
def method_not_allowed(e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "请求方法不允许。"}), 405
    return render_template("404.html"), 405


@app.errorhandler(500)
def internal_error(e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "服务器内部错误。"}), 500
    return render_template("404.html"), 500


# ---------------------------------------------------------------------------
# 拓展A：导出 CSV（第7天保留）
# ---------------------------------------------------------------------------

@app.route("/download")
@login_required
def download():
    # 1. 获取当前选择的品类参数（默认为"全部"）
    category = request.args.get("category", "全部")

    # 2. 调用数据服务，获取对应品类的数据
    data = load_dashboard_data(BASE_DIR, category)

    # 3. 在内存中生成 CSV 文件流
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(["偏好品类", "用户数", "流失率", "平均订单数"])
    for row in data["category_rows"]:
        cw.writerow([row["偏好品类"], row["用户数"], row["流失率"], row["平均订单数"]])

    # 4. 将字符串流转换为字节流，并将指针移到开头
    output = io.BytesIO(si.getvalue().encode("utf-8-sig"))
    output.seek(0)

    # 5. 动态设置文件名并触发浏览器下载
    filename = f"export_{category}.csv" if category != "全部" else "export_all.csv"
    return send_file(output, mimetype="text/csv", download_name=filename, as_attachment=True)


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, port=5000)
