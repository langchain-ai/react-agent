"""Sample ticket data for the mock database."""

SAMPLE_TICKETS = [
    ('12345', 'Subject: Delivery Issue - Order #ZD78932\n\nHi there,\nI placed an order on May 10th, 2023 (Order #ZD78932) but haven\'t received my package yet. The tracking information hasn\'t updated in 3 days. Can you please help me figure out what\'s happening with my delivery?\n\nThanks,\nJamie Smith'),
    ('14983', 'Subject: Address Verification Needed - Desk Chair Order\n\nHello,\nI recently ordered a desk chair from your store (Order #ZD81045). I received an email saying my shipping address needs verification before you can process my order. Please let me know what information you need from me.\n\nBest regards,\nAlex Johnson'),
    ('23456', 'Subject: Damaged Product - Refund Request\n\nDear Support Team,\nI received my order yesterday but unfortunately the item is damaged. I\'ve attached photos showing the broken parts. I would like to request a full refund as the product is unusable in this condition.\n\nRegards,\nTaylor Williams'),
    ('34567', 'Subject: Wrong Color Sent - Exchange Request\n\nHi,\nI ordered the premium headphones in black, but received them in white instead. I\'d like to exchange them for the color I originally ordered. Please advise on the exchange process.\n\nThank you,\nMorgan Lee'),
    ('45678', 'Subject: Severe Delivery Delay - Order #ZD92103\n\nTo whom it may concern,\nMy order is now 5 days past the estimated delivery date you provided. This is unacceptable as I needed these items for an event this weekend. Please expedite my shipment or provide compensation for this inconvenience.\n\nDisappointed,\nJordan Rivera'),
    ('56789', 'Subject: Warranty Question - Purchase from 11 months ago\n\nHello Support,\nI purchased your premium blender about 11 months ago and it\'s starting to make strange noises. I believe it should still be under warranty. Can you please clarify what\'s covered and how I can get it repaired?\n\nThanks,\nCasey Thompson'),
    ('67890', 'Subject: Bulk Purchase Inquiry - Corporate Order\n\nHello,\nI\'m interested in placing a bulk order of your office chairs for our company. We need approximately 50 units. Could you provide information about bulk pricing, customization options, and estimated delivery timeframes for large orders?\n\nBest,\nSam Wilson\nProcurement Manager'),
    ('78901', 'Subject: Assembly Help Needed - Can\'t Follow Instructions\n\nHi there,\nI\'ve been trying to assemble the bookshelf I purchased but I\'m completely stuck. The instructions are confusing and some parts don\'t seem to fit together as shown. Can someone please help me with this? I\'ve spent hours trying to figure it out.\n\nFrustrated,\nRiley Garcia'),
    ('89012', 'Subject: Positive Feedback - Recent Purchase\n\nHello Team,\nI just wanted to let you know how pleased I am with my recent purchase. The quality exceeds my expectations and the delivery was faster than anticipated. Your customer service was excellent throughout the process. I\'ll definitely be shopping with you again!\n\nThank you,\nAvery Martinez'),
    ('90123', 'Subject: Product Availability Inquiry\n\nHi,\nI\'ve been trying to purchase the ergonomic desk chair (model #EC-420) from your website, but it shows as out of stock. Do you know when this item will be available again? I\'d really prefer this specific model and would be willing to pre-order if possible.\n\nRegards,\nDrew Peterson')
]

# Sample comments for tickets
SAMPLE_COMMENTS = [
    ('12345', 'Agent: Thank you for contacting us about your delivery issue. I can see your order in our system and will investigate what\'s happening with the shipment. Could you confirm the last four digits of your phone number for verification?'),
    ('12345', 'Agent: I\'ve checked with our shipping department and confirmed your delivery details. It appears there was a delay at our distribution center. I apologize for the inconvenience.'),
    ('12345', 'Agent: Good news! I\'ve arranged for your delivery to be prioritized. You should receive your package by next Wednesday. We\'ve also added a 10% discount to your account for your next purchase due to this inconvenience.'),
    ('23456', 'Agent: I\'ve received the photos of your damaged product. Thank you for providing such clear images of the issue. I\'ll escalate this to our returns department right away.'),
    ('23456', 'Agent: I\'m pleased to inform you that your refund request has been approved. The full amount will be credited back to your original payment method within 3-5 business days. Is there anything else I can assist you with today?'),
    ('34567', 'Agent: I\'ve reviewed your order and can confirm that the wrong color was indeed shipped. I sincerely apologize for this error on our part.'),
    ('34567', 'Agent: I\'ve arranged for a return shipping label to be emailed to you. Once you receive it, please package the item and drop it off at any carrier location. No need to include the original packaging.'),
    ('34567', 'Agent: Great news! Your replacement headphones in black have been shipped today. You should receive them within 2-3 business days. The tracking number has been sent to your email.'),
    ('45678', 'Agent: I sincerely apologize for the significant delay with your order. This falls well below our service standards, and I understand your frustration completely.'),
    ('45678', 'Agent: I\'ve spoken with our logistics team and have arranged for your order to be expedited with priority shipping. Additionally, we\'ll be refunding your shipping costs and adding a $20 store credit to your account for the inconvenience caused.'),
    ('56789', 'Agent: Thank you for your inquiry about your blender\'s warranty. I\'m happy to inform you that our premium blenders come with a 24-month warranty that covers manufacturing defects and mechanical failures during normal use.'),
    ('67890', 'Agent: Thank you for your interest in a bulk purchase. I\'ve prepared a detailed quote for 50 office chairs, including a 15% bulk discount. For orders of this size, we can offer custom color options and logo embroidery. Typical delivery for bulk orders is 3-4 weeks, but we can discuss expedited options if needed.'),
    ('78901', 'Agent: I\'m sorry to hear you\'re having trouble with assembly. I\'ve sent a link to a video tutorial that should help clarify the steps. If you\'re still having issues after watching it, we can arrange for a video call with one of our assembly specialists.'),
    ('89012', 'Agent: Thank you so much for taking the time to share your positive experience! We\'re thrilled to hear that you\'re happy with both the product and our service. Your feedback has been shared with our team, and we look forward to serving you again in the future.'),
    ('90123', 'Agent: Thank you for your interest in our ergonomic desk chair. I\'ve checked our inventory system, and I\'m happy to inform you that we expect a new shipment to arrive on the 15th of next month. Would you like me to set up a notification to alert you when it\'s back in stock?')
]

# Sample addresses for tickets
SAMPLE_ADDRESSES = [
    ('12345', 'Bahnhofstrasse 42, 8001 Zurich, Switzerland'),
    ('14983', 'Heinrichstrasse 267, 8005 Zurich, Switzerland'),
    ('23456', 'Rämistrasse 101, 8092 Zurich, Switzerland'),
    ('34567', 'Langstrasse 14, 8004 Zurich, Switzerland'),
    ('45678', 'Europaallee 21, 8004 Zurich, Switzerland'),
    ('56789', 'Hardbrücke 30, 8005 Zurich, Switzerland'),
    ('67890', 'Limmatstrasse 118, 8005 Zurich, Switzerland'),
    ('78901', 'Josefstrasse 52, 8005 Zurich, Switzerland'),
    ('89012', 'Badenerstrasse 120, 8004 Zurich, Switzerland'),
    ('90123', 'Stauffacherstrasse 60, 8004 Zurich, Switzerland')
]
