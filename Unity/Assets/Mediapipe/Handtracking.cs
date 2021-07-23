using System;

using UnityEngine;
using UnityEngine.Events;

namespace mediapipe.wrapper
{
    public class Handtracking : IDisposable
    {
        public readonly DataEvent OnData = new DataEvent();
        private AndroidJavaObject m_Instance;

        public Handtracking(int num_hands = 2) {
            using (AndroidJavaClass unityPlayer = new AndroidJavaClass("com.unity3d.player.UnityPlayer")) {
                using (AndroidJavaObject activity = unityPlayer.GetStatic<AndroidJavaObject>("currentActivity")) {
                    m_Instance = new AndroidJavaObject("com.rrrfffrrr.unity.mediapipe.handtracking.Graph", activity, new DataCallback(data: DataHandler), num_hands);
                }
            }
        }

        public void Update(Texture texture) {
            var tex = texture.GetNativeTexturePtr().ToInt32();
            m_Instance?.Call("Input", tex, texture.width, texture.height);
        }

        public void Dispose() { 
            m_Instance?.Call("Close");
            m_Instance?.Dispose();
        }

        private void DataHandler(TrackingData data) {
            OnData.Invoke(data);
        }

        [Serializable] public class TrackingData
        {
            public long timestamp;
            public Hand[] hands;

            [Serializable] public class Hand
            {
                public Landmark[] landmarks;

                [Serializable] public class Landmark
                {
                    public float x, y, z;
                    public float visibility;
                    public float presence;
                }
            }
        }

        [Serializable] public class DataEvent : UnityEvent<TrackingData> { }
        public class DataCallback : AndroidJavaProxy
        {
            public delegate void ReceiveAction(TrackingData data);

            private readonly ReceiveAction OnDataCallback;

            public DataCallback(ReceiveAction data) : base("com.rrrfffrrr.unity.mediapipe.handtracking.DataCallback") {
                OnDataCallback = data;
            }

            [System.Diagnostics.CodeAnalysis.SuppressMessage("Style", "IDE1006", Justification = "<Kotlin code convention>")]
            public void onData(string data) {
                OnDataCallback?.Invoke(JsonUtility.FromJson<TrackingData>(data));
            }
        }

    }
}