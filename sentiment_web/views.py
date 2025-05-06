
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
from django.db.models import Count
from django.utils.timezone import now, timedelta
import cv2
import os
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image
from torchvision import transforms
import random
from django.core.mail import send_mail
from django.core.cache import cache
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import FeatureUsage, PageView

from functools import wraps
from django.contrib.auth.decorators import user_passes_test

def admin_required(view_func):
    def _wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return redirect(f'/admin_login/?next={request.path}')
        if not request.user.is_superuser:
            return redirect('/')  # 普通用户直接踢回首页
        return view_func(request, *args, **kwargs)
    return _wrapped_view

def record_page_view_auto(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        PageView.objects.create(
            user=request.user if request.user.is_authenticated else None,
            page_name=request.path,  # 自动使用访问的URL路径
            ip_address=request.META.get('REMOTE_ADDR')
        )
        return view_func(request, *args, **kwargs)
    return wrapper


# 加载word模型和tokenizer
model = BertForSequenceClassification.from_pretrained('./final_model')
tokenizer = BertTokenizer.from_pretrained('./final_model')


# vision model
model_vision = torch.load("sentiment_web/model-64-unpretrained.pth", map_location=torch.device('cpu'),weights_only=False)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一图像大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])
model_vision.eval()  # 设置为推理模式


def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    softmax = torch.nn.Softmax(dim=-1)
    probabilities = softmax(logits)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    positive_prob = probabilities[0][1].item()  # 假设类别1是积极情绪
    negative_prob = probabilities[0][0].item()  # 假设类别0是消极情绪
    cla="Positive" if predicted_class == 1 else "Negative"
    return cla,positive_prob,negative_prob


@record_page_view_auto
@login_required()
def face_detection(request):
    if request.method == 'POST' and request.FILES.get('image'):
        FeatureUsage.objects.create(user=request.user, feature_name='visual')


        image = request.FILES['image']
        name=image.name
        image_path = os.path.join(settings.MEDIA_ROOT, name)

        # 保存原始图片
        with default_storage.open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        # 读取图片并进行人脸检测
        import dlib
        detector = dlib.get_frontal_face_detector()
        # 读取图像
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orin_h, orin_w, channels = img.shape

        # 使用 MTCNN 检测人脸
        faces = detector(gray)

        # 绘制检测到的人脸框
        if not faces:
            return render(request, 'result.html', {'error': '未检测到人脸，请上传清晰的人脸图片！'})
        face = faces[0]
        x, y, width, height = (face.left(), face.top(), face.width(), face.height())
        if (width)/(orin_w) > 0.8:
            face=img
        else:
            print(1)
            face = img[y-int(height/6):y + height+int(height/3), x-int(width/6):x + width+int(width/3)]
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # 转换为 PIL 格式
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image = transform(face_pil).unsqueeze(0).to(device)  # 增加 batch 维度
        # 进行预测
        with torch.no_grad():
            output = model_vision(image)
            _, predicted = torch.max(output, 1)  # 获取最大概率的类别索引

        # 你的类别标签
        class_labels = ["happy", "sad", "natrual"]  # 修改为你的实际类别
        emotion = class_labels[predicted.item()]
        print(emotion)

        # --- 打标签 ---
        font_scale = 1.2  # 字体缩放大小
        font_thickness = 2  # 字体粗细
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size, _ = cv2.getTextSize(emotion, font, font_scale, font_thickness)
        text_width, text_height = text_size

        # 标签位置固定在右上角，稍微离边界留一点空隙
        text_x = orin_w - text_width - 20
        text_y = 40  # 距离上方一点点

        # 先画背景矩形框，让字更清楚
        cv2.rectangle(
            img,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + 5),
            (0, 0, 0),
            thickness=-1
        )
        # 再画文字
        cv2.putText(img, emotion, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

        # 保存处理后的图片
        processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_' + name)
        cv2.imwrite(processed_image_path, img)

        return render(request, 'result.html', {'image_url': settings.MEDIA_URL + 'processed_' + name})

    return render(request, 'upload.html')

@record_page_view_auto
@login_required(login_url='/login/')
def combined_sentiment(request):
    sentiment = None
    text_sentiment_class = None
    image_sentiment_class = None
    combined_score = None
    text = None

    if request.method == 'POST' and request.FILES.get('image') and request.POST.get('text'):
        FeatureUsage.objects.create(user=request.user, feature_name='combined')
        # 处理文本
        text = request.POST.get('text')
        text_class, text_positive_prob, text_negative_prob = predict_sentiment(text)

        # 处理图像
        image = request.FILES['image']
        name = image.name
        image_path = os.path.join(settings.MEDIA_ROOT, name)

        # 保存原始图片
        with default_storage.open(image_path, 'wb+') as destination:
            for chunk in image.chunks():
                destination.write(chunk)

        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = img.to(device)
        model_vision.to(device)
        model_vision.eval()

        with torch.no_grad():
            outputs = model_vision(img)
            probabilities = nn.Softmax(dim=1)(outputs)
            image_positive_prob = probabilities[0][0].item()
            image_negative_prob = probabilities[0][1].item()

        # 加权计算
        final_positive_score = 0.7 * text_positive_prob + 0.3 * image_positive_prob
        final_negative_score = 0.7 * text_negative_prob + 0.3 * image_negative_prob

        combined_score = (final_positive_score, final_negative_score)
        if final_positive_score >= final_negative_score:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        text_sentiment_class = text_class
        image_sentiment_class = "Positive" if image_positive_prob >= image_negative_prob else "Negative"

        # 保存一条功能使用记录
        FeatureUsage.objects.create(user=request.user, feature_name='combined')

    return render(request, 'combined_sentiment.html', {
        'sentiment': sentiment,
        'text_sentiment_class': text_sentiment_class,
        'image_sentiment_class': image_sentiment_class,
        'combined_score': combined_score,
        'text': text,
    })

@record_page_view_auto
@login_required()
def text_sentiment(request):
    sentiment = None
    positive_prob = None
    negative_prob = None
    text = None

    if request.method == 'POST':
        FeatureUsage.objects.create(user=request.user, feature_name='text')
        text = request.POST.get('text')
        sentiment, positive_prob, negative_prob = predict_sentiment(text)

    return render(request, 'index.html', {
        'sentiment': sentiment,
        'text': text,
        'positive_prob': positive_prob,
        'negative_prob': negative_prob,
    })


def api_predict(request):
    if request.method == "POST":
        text = request.POST.get("text")
        sentiment = predict_sentiment(text)
        return JsonResponse({"sentiment": sentiment})


@record_page_view_auto
@csrf_exempt
def send_code(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        code = ''.join(random.choices('0123456789', k=6))
        cache.set(f'verify_{email}', code, timeout=300)
        send_mail('验证码', f'您的验证码是：{code}', None, [email])
        return JsonResponse({'status': 'ok'})
    return JsonResponse({'status': 'error'})

@record_page_view_auto
@csrf_exempt
def register(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        code = request.POST.get('code')
        real_code = cache.get(f'verify_{email}')
        if code != real_code:
            return JsonResponse({'status': 'error', 'msg': '验证码错误'})
        if User.objects.filter(username=email).exists():
            return JsonResponse({'status': 'error', 'msg': '用户已存在'})
        User.objects.create_user(username=email, email=email, password=password)
        return JsonResponse({'status': 'ok', 'msg': '注册成功'})

    # 处理 GET 请求，返回注册页面
    return render(request, 'register.html')

@record_page_view_auto
def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            next_url = request.GET.get('next', '/home')  # 登录成功后跳转
            return redirect(next_url)
        else:
            return render(request, 'login.html', {'error': '用户名或密码错误'})

    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect('login')

@login_required(login_url='/login/')
def home(request):
    return render(request, 'home.html')

@admin_required
def admin_dashboard(request):
    user_count = User.objects.count()
    feature_usage_count = FeatureUsage.objects.count()
    page_view_count = PageView.objects.count()
    return render(request, 'admin_panel/dashboard.html', {
        'user_count': user_count,
        'feature_usage_count': feature_usage_count,
        'page_view_count': page_view_count,
    })

@admin_required
def user_list(request):
    users = User.objects.filter(is_superuser=False)  # 只查普通用户
    return render(request, 'admin_panel/user_list.html', {'users': users})

@admin_required
def feature_usage(request):
    feature_data = FeatureUsage.objects.values('feature_name').annotate(count=Count('id'))
    return render(request, 'admin_panel/feature_usage.html', {'feature_data': feature_data})

@admin_required
def page_views(request):
    # 获取最近7天访问数据
    today = now().date()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    pageview_data = []
    for day in days:
        count = PageView.objects.filter(timestamp__date=day).count()
        pageview_data.append({'date': day.strftime('%Y-%m-%d'), 'count': count})

    return render(request, 'admin_panel/page_views.html', {'pageview_data': pageview_data})

from django.contrib.auth.models import User
from django.http import JsonResponse

@admin_required
@csrf_exempt
def disable_user(request, user_id):
    try:
        user = User.objects.get(pk=user_id)
        user.is_active = False
        user.save()
        return JsonResponse({'status': 'ok'})
    except User.DoesNotExist:
        return JsonResponse({'status': 'error', 'msg': '用户不存在'})

@admin_required
@csrf_exempt
def delete_user(request, user_id):
    try:
        user = User.objects.get(pk=user_id)
        user.delete()
        return JsonResponse({'status': 'ok'})
    except User.DoesNotExist:
        return JsonResponse({'status': 'error', 'msg': '用户不存在'})


from django.core.cache import cache
from django.core.mail import send_mail
import random


@csrf_exempt
def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        code = request.POST.get('code')
        try:
            # 先用 email 找到真正的 username
            user_obj = User.objects.filter(email=username).first()
            username = user_obj.username
        except User.DoesNotExist:
            return render(request, 'admin_panel/admin_login.html', {'error': '邮箱不存在'})
        user = authenticate(request, username=username, password=password)

        if user and user.is_superuser:
            real_code = cache.get(f'verify_{user.email}')
            if real_code is None:
                return render(request, 'admin_panel/admin_login.html', {'error': '请先发送验证码'})
            if code != real_code:
                return render(request, 'admin_panel/admin_login.html', {'error': '验证码错误'})
            # 验证码正确
            login(request, user)
            return redirect('/admin_panel/')
        else:
            return render(request, 'admin_panel/admin_login.html', {'error': '用户名或密码错误，或者没有管理员权限'})

    return render(request, 'admin_panel/admin_login.html')


