import cv2
import numpy as np

# --- Colors ---
MAROON = (0, 0, 128)      
RED_ALERT = (0, 0, 255)  
ZONE_COLOR = (0, 0, 255)  

def check_trespassing(bbox, zone_coords):
    x1, y1, x2, y2 = bbox
    # Check if the "feet" are in the zone
    foot_x, foot_y = int((x1 + x2) / 2), int(y2)
    zx1, zy1, zx2, zy2 = zone_coords
    return zx1 < foot_x < zx2 and zy1 < foot_y < zy2

def check_loitering(track_id, center_point, track_history, current_time, threshold):
    if track_id not in track_history:
        track_history[track_id].append((current_time, center_point))
        return False
    
    first_time = track_history[track_id][0][0]
    duration = current_time - first_time
    track_history[track_id].append((current_time, center_point))

    if duration > threshold:
        return True
    return False

def process_frame_annotations(frame, tracks, current_time, track_history, loitering_saved, settings):
    annotated_frame = frame.copy()
    
    # 1. Draw Transparent Restricted Zone
    if settings['trespassing_enabled']:
        tz = settings['trespassing_zone']
        
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (tz[0], tz[1]), (tz[2], tz[3]), ZONE_COLOR, -1)
        
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
        
        cv2.rectangle(annotated_frame, (tz[0], tz[1]), (tz[2], tz[3]), MAROON, 2)
        cv2.putText(annotated_frame, "Restricted Zone", (tz[0], tz[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAROON, 2)

    frame_alerts = {
        'count': 0,
        'trespassing': False,
        'loitering': False,
        'crowd': False
    }

    for track in tracks:
        # Standard filter to avoid ghost tracks
        if not track.is_confirmed() and track.time_since_update > 1:
            continue
        
        frame_alerts['count'] += 1
        track_id = track.track_id
        
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        bbox = (x1, y1, x2, y2)
        center = (int((x1+x2)/2), int((y1+y2)/2))

        # Check Trespassing
        if settings['trespassing_enabled'] and check_trespassing(bbox, settings['trespassing_zone']):
            frame_alerts['trespassing'] = True

        # Check Loitering
        if settings['loitering_enabled']:
            if check_loitering(track_id, center, track_history, current_time, settings['loitering_threshold']):
                frame_alerts['loitering'] = True
                loitering_saved[track_id] = True
    
    #check crowd
    if settings['crowd_enabled'] and frame_alerts['count'] > settings['crowd_threshold']:
        frame_alerts['crowd'] = True


    y_pos = 20
    cv2.putText(annotated_frame, f"People Count: {frame_alerts['count']}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAROON, 2)
    
    if frame_alerts['crowd']:
        y_pos += 20
        cv2.putText(annotated_frame, "Crowd Alert!", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_ALERT, 2)

    if frame_alerts['loitering']:
        y_pos += 20
        cv2.putText(annotated_frame, "Loitering Alert!", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_ALERT, 2)

    if frame_alerts['trespassing']:
        y_pos += 20
        cv2.putText(annotated_frame, "Trespassing Alert!", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED_ALERT, 2)

    return annotated_frame, frame_alerts