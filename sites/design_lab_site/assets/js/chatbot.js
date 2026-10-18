document.addEventListener("DOMContentLoaded", function () {
    const toggler = document.querySelector(".chatbot-toggler");
    const windowEl = document.querySelector(".chatbot-window");
    const closeBtn = document.querySelector(".chat-close");
    const chatBody = document.querySelector(".chat-body");
    const chatInput = document.querySelector(".chat-input");
    const sendBtn = document.querySelector(".chat-send");

    const scrollToBottom = () => {
        setTimeout(() => chatBody.scrollTop = chatBody.scrollHeight + 500, 50);
    };

    toggler.addEventListener("click", () => {
        windowEl.classList.toggle("show");
        if (windowEl.classList.contains("show")) {
            chatInput.focus();
            scrollToBottom();
        }
    });
    closeBtn.addEventListener("click", () => windowEl.classList.remove("show"));

    document.addEventListener("keydown", e => {
        if (e.key === "Escape" && windowEl.classList.contains("show")) windowEl.classList.remove("show");
    });

    document.addEventListener("click", e => {
        if (!windowEl.classList.contains("show")) return;
        const hoverArea = document.querySelector(".chatbot-hover-area");
        const inside = windowEl.contains(e.target) || toggler.contains(e.target) || (hoverArea && hoverArea.contains(e.target));
        if (!inside) windowEl.classList.remove("show");
    });

    const addMessage = (text, sender) => {
        const div = document.createElement("div");
        div.className = `chat-message ${sender}`;
        div.innerHTML = text;
        chatBody.appendChild(div);
        scrollToBottom();
    };

    let typingIndicator = null;
    const showTyping = () => {
        if (typingIndicator) typingIndicator.remove();
        typingIndicator = document.createElement("div");
        typingIndicator.className = "chat-message bot typing-indicator";
        typingIndicator.innerHTML = `
            <i class="fas fa-ellipsis-h typing-dots"></i>
            <i class="fas fa-ellipsis-h typing-dots"></i>
            <i class="fas fa-ellipsis-h typing-dots"></i>
            <span style="font-style:italic; margin-left:8px; color:#94a3b8;">Laira печатает…</span>
        `;
        chatBody.appendChild(typingIndicator);
        scrollToBottom();
    };

    const hideTyping = () => {
        if (typingIndicator) {
            typingIndicator.style.opacity = "0";
            setTimeout(() => typingIndicator?.remove(), 300);
            typingIndicator = null;
        }
    };

    // ─────────────────────── LAIRA — ИДЕАЛЬНАЯ ЛЕДИ 2025 ───────────────────────
    const botReply = (text) => {
        let msg = text.toLowerCase().trim()
            .replace(/[.,!?\n\r;:—–()«»"]/g, ' ')
            .replace(/\s+/g, ' ');

        const normalize = (s) => s
            // Приветствия
            .replace(/прив+/g, 'привет')
            .replace(/здаров+/g, 'привет')
            .replace(/дарoва+/g, 'привет')
            .replace(/хи+|хай+|хеллоу+|хелло|hello|hi|hey/g, 'привет')
            .replace(/ку+|qq/g, 'привет')
            .replace(/здрасте|здрастуйте|добрый день|доброе утро|добрый вечер/g, 'привет')
            // Цены и деньги
            .replace(/скок+/g, 'сколько')
            .replace(/ценник+|прайс+|тариф+|бюджет+/g, 'цена')
            // Контакты
            .replace(/тг|телег|telegram|tg/g, 'телеграм')
            .replace(/почта|мыло|email|e-mail|имейл/g, 'почта')
            // Кейсы
            .replace(/кейс+|case+|пример+/g, 'кейсы')
            // Благодарность и прощание
            .replace(/спс|спасибк|спасибочки/g, 'спасибо')
            .replace(/покеда|покаа+/g, 'пока');

        msg = normalize(msg);

        let reply = "";

        // 1. Приветствия (20 вариантов)
        if (/привет|здравствуй|добрый|здравствуйте/i.test(msg)) {
            const greets = [
                "Здравствуйте! Меня зовут Laira. Очень рада нашему знакомству.",
                "Добрый день! Спасибо, что заглянули. Чем могу быть полезна?",
                "Здравствуйте! С удовольствием помогу вам с любым вопросом.",
                "Приветствую вас! Надеюсь, ваш день проходит замечательно.",
                "Добрый день! Я здесь, чтобы ответить на все интересующие вас вопросы.",
                "Здравствуйте! Как приятно, что вы написали.",
                "Приветствую! Готова помочь по всем направлениям нашей работы.",
                "Добрый день! Рада видеть вас на нашем сайте.",
                "Здравствуйте! Буду рада нашему диалогу.",
                "Приветствую вас! Чем могу быть полезна сегодня?",
                "Добрый день! Благодарю за ваше обращение.",
                "Здравствуйте! С радостью отвечу на любые вопросы.",
                "Приветствую! Очень приятно познакомиться.",
                "Добрый день! Жду вашего вопроса с нетерпением.",
                "Здравствуйте! Спасибо, что нашли время заглянуть к нам.",
                "Приветствую вас! Надеюсь, смогу быть максимально полезной.",
                "Добрый день! Рада приветствовать вас.",
                "Здравствуйте! Искренне рада нашему общению.",
                "Приветствую! Как могу помочь вам сегодня?",
                "Добрый день! Благодарю за внимание к нашей студии."
            ];
            reply = greets[Math.floor(Math.random() * greets.length)];
        }

        // 2. Как дела? (12 вариантов)
        else if (/как дела|как вы|как поживаете|как самочувствие|как настроение/i.test(msg)) {
            const answers = [
                "Благодарю вас, всё прекрасно. А как ваши дела?",
                "Спасибо, у меня всё замечательно. Надеюсь, у вас тоже.",
                "Прекрасно, благодарю за внимание. Как вы себя чувствуете?",
                "Очень хорошо, спасибо. Рада нашему общению.",
                "Благодарю, всё отлично. А как дела у вас?",
                "Спасибо, прекрасно. Желаю вам отличного дня.",
                "У меня всё замечательно, благодарю. Как ваши дела?",
                "Прекрасно, спасибо. Очень приятно, что вы спросили.",
                "Благодарю вас, всё хорошо. Надеюсь, у вас тоже.",
                "Спасибо, прекрасно. Как ваше настроение?",
                "Всё отлично, благодарю. А как вы?",
                "Прекрасно, спасибо за заботу. Как ваши дела?"
            ];
            reply = answers[Math.floor(Math.random() * answers.length)];
        }

        // 3. Представление (10 вариантов)
        else if (/кто вы|кто ты|представьтесь|как вас зовут|что за лайра/i.test(msg)) {
            const intros = [
                "Меня зовут Laira. Я — виртуальный ассистент студии Design Lab, специализирующейся на UX/UI-дизайне для FinTech и LegalTech.",
                "Я Laira — ваш помощник. С радостью расскажу об услугах, ценах, кейсах и способах связи.",
                "Приятно представить: я Laira, ассистент Design Lab. Готова ответить на любые вопросы.",
                "Меня зовут Laira. Помогаю гостям сайта сориентироваться в наших возможностях.",
                "Я — Laira, ваш интеллектуальный ассистент. Чем могу быть полезна?",
                "Здравствуйте! Я Laira — помощница студии Design Lab. Рада помочь.",
                "Меня зовут Laira. Я здесь, чтобы сделать ваше знакомство с нами максимально комфортным.",
                "Я Laira — ваш вежливый гид по услугам и проектам Design Lab.",
                "Приветствую! Меня зовут Laira, я ваш ассистент.",
                "Я Laira. Создана помогать вам узнать всё о нашей студии."
            ];
            reply = intros[Math.floor(Math.random() * intros.length)];
        }

        // 4. Услуги
        else if (/услуг|что вы делаете|чем занимаетесь|дизайн|ui|ux|финтех|легалтех|саас|веб|мобильн/i.test(msg)) {
            reply = `Мы — Design Lab, создаём продуманный и элегантный UX/UI-дизайн для сложных продуктов:<br><br>
            • FinTech и банковские сервисы<br>
            • LegalTech и юридические платформы<br>
            • SaaS-решения и личные кабинеты<br>
            • Корпоративные сайты и лендинги<br><br>
            Подробности → <a href="services.html" target="_blank">Услуги</a>`;
        }

        // 5. Кейсы
        else if (/портфолио|кейс|пример|работы|проекты|покажите|cases/i.test(msg)) {
            reply = `С большой радостью покажу наши проекты.<br><br>
            <a href="cases.html" target="_blank">Все кейсы здесь → cases.html</a><br><br>
            Там вы увидите работы для банков, юридических сервисов и сложных веб-приложений.`;
        }

        // 6. Цены
        else if (/цена|стоимост|сколько|прайс|бюджет|от скольки/i.test(msg)) {
            reply = `Ориентировочные цены на 2025 год:<br><br>
            • Корпоративный сайт / лендинг — <b>от 250 000 ₽</b><br>
            • Веб-приложение (SaaS, кабинет) — <b>от 800 000 ₽</b><br>
            • Мобильное приложение — <b>от 1 200 000 ₽</b><br>
            • Полный редизайн — <b>от 1 500 000 ₽</b><br>
            • Аудит интерфейсов — <b>150 000 ₽</b><br><br>
            Точную стоимость определим после обсуждения задач.`;
        }

        // 7. Сроки
        else if (/срок|когда|как быстро|дедлайн|тайминг|сколько времени/i.test(msg)) {
            reply = `Ориентировочные сроки реализации:<br><br>
            • Лендинг — 3–6 недель<br>
            • Веб-приложение — 2–5 месяцев<br>
            • Мобильное приложение — 3–7 месяцев<br>
            • Аудит — 7–14 дней<br><br>
            Готовы обсудить индивидуальный график.`;
        }

        // 8. Контакты
        else if (/контакт|телеграм|почта|связь|куда писать|email/i.test(msg)) {
            reply = `Буду искренне рада вашему сообщению:<br><br>
            • Telegram: <a href="https://t.me/datalaw_rights" target="_blank">@datalaw_rights</a> (самый быстрый способ)<br>
            • Email: <a href="mailto:1448884@gmail.com">1448884@gmail.com</a><br><br>
            Отвечаем в течение 1–2 часов в рабочее время (МСК).`;
        }

        // 9. Стиль сайта
        else if (/стиль|neumorphism|тёмный|dark|почему тёмный|красиво/i.test(msg)) {
            reply = "Это наш фирменный стиль Dark Neumorphism. Он разработан для максимального комфорта глаз при длительной работе и создаёт ощущение современности и элегантности. Очень рада, если он вам понравился.";
        }

        // 10. Комплименты
        else if (/красиво|классно|нравится|отлично|прекрасно|элегантно|стильн|великолепно/i.test(msg)) {
            const comp = [
                "Благодарю вас от всего сердца! Такие слова очень вдохновляют.",
                "Как приятно это слышать! Спасибо вам большое.",
                "Искренне рада, что вам понравилось. Это лучшая награда для нас.",
                "Спасибо! Мы действительно старались создать что-то особенное.",
                "Очень тронута вашими словами. Благодарю!",
                "Как замечательно, что наш стиль пришёлся вам по душе. Спасибо!"
            ];
            reply = comp[Math.floor(Math.random() * comp.length)];
        }

        // 11. Спасибо
        else if (/спасиб|благодарю|благодарна/i.test(msg)) {
            reply = "Пожалуйста! Всегда рада быть полезной. Обращайтесь в любое время.";
        }

        // 12. До свидания
        else if (/пока|до свидания|до встречи|всего доброго|хорошего дня/i.test(msg)) {
            reply = "До свидания! Желаю вам прекрасного дня и жду новых встреч.";
        }

        // 13. Не поняла — только идеальная вежливость
        else {
            const fallback = [
                "Прошу прощения, я не вполне поняла ваш вопрос. Не могли бы вы уточнить, пожалуйста?",
                "Благодарю за сообщение. К сожалению, я не совсем уловил суть. Буду признательна за пояснение.",
                "Простите, я немного растерялась. Можете переформулировать, пожалуйста?",
                "Прошу прощения, я не совсем поняла. Буду очень благодарна, если вы уточните.",
                "К сожалению, я не вполне разобрала ваш вопрос. Помогите мне, пожалуйста, понять вас лучше.",
                "Благодарю за обращение. Не могли бы вы немного пояснить, о чём идёт речь?"
            ];
            reply = fallback[Math.floor(Math.random() * fallback.length)];
        }

        showTyping();
        setTimeout(() => {
            hideTyping();
            addMessage(reply, "bot");
        }, 1200 + Math.random() * 1000);
    };

    const handleSend = () => {
        const text = chatInput.value.trim();
        if (!text) return;
        addMessage(text, "user");
        chatInput.value = "";
        botReply(text);
    };

    sendBtn.addEventListener("click", handleSend);
    chatInput.addEventListener("keypress", e => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Уведомления и анимации (без изменений)
    const orderForm = document.getElementById("orderForm");
    const notificationContainer = document.createElement("div");
    notificationContainer.style.cssText = "position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:1050;";
    document.body.appendChild(notificationContainer);

    const showNotification = (msg, type = 'info') => {
        const n = document.createElement("div");
        n.className = `alert alert-${type} alert-dismissible fade show`;
        n.style.cssText = "background:var(--bg-element);color:var(--text-main);border-radius:12px;border:1px solid rgba(255,255,255,.1);box-shadow:0 10px 20px rgba(0,0,0,.4);margin-top:10px;max-width:300px;text-align:center;";
        n.innerHTML = `${msg}<button type="button" class="btn-close" data-bs-dismiss="alert" style="filter:invert(1);"></button>`;
        notificationContainer.appendChild(n);
        setTimeout(() => { n.classList.remove('show'); n.classList.add('fade'); setTimeout(() => n.remove(), 400); }, 5000);
    };

    if (orderForm) {
        orderForm.addEventListener("submit", e => {
            e.preventDefault();
            showNotification("Заявка успешно отправлена. Благодарю за обращение!", 'success');
            orderForm.reset();
        });
    }

    const els = document.querySelectorAll('[data-animate="fade-in-up"]');
    els.forEach(el => el.style.opacity = "0");
    const check = () => {
        els.forEach(el => {
            if (el.getBoundingClientRect().top < window.innerHeight - 50) {
                el.style.opacity = "1";
                el.style.transform = "translateY(0)";
            }
        });
    };
    window.addEventListener('scroll', check);
    check();

    window.addEventListener('load', () => {
        window.scrollTo(0, 0);
        document.body.classList.add('loaded');
        if (typeof AOS !== 'undefined') AOS.init({ duration: 800, once: true });
    });
});