@echo off
echo ========================================
echo  Trifeet V2 系统性全量训练
echo  异步分布式: CPU 自对弈 + GPU 训练
echo  课程学习 + 王座模式 (KOTH)
echo ========================================
echo.
echo 参数: --resume --actors 8 --auto-curriculum --koth-mode
echo 监控: 打开 train_dashboard.html 连接 ws://localhost:5001
echo.
python -m ai.train_async --resume --actors 8 --auto-curriculum --koth-mode
pause
